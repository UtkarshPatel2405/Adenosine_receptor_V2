"""Master prediction API coordinating inference, pharmacology, 3D interaction, safety, and ADMET."""
from typing import Dict, Any, List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from src.config import SUBTYPES, MODELS_DIR
from src.features import build_features
from src.applicability_domain import check_applicability_domain
from src.models.model_loader import (
    _load_scaler, _load_xgb_models, _load_lgb_models,
    _load_rf_models, _load_stack_models, _load_db_lookup
)
from src.models.ensemble_engine import _ensemble_predict, _predict_one_subtype, _try_gnn_predict, _ZERO_RESULT
from src.models.selectivity_engine import compute_selectivity_spectrum
from src.models.efficacy_engine import predict_functional_efficacy
from src.models.interaction_engine import analyze_pocket_interactions
from src.models.safety_engine import evaluate_safety_profile
from src.models.admet_engine import evaluate_cns_admet
from src.models.adaptive_conformal import calibrate_adaptive_interval
from src.models.multitask_covariance import pchembl_to_ki_nm, format_ki_display, regularize_multitask_predictions


def _descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    """Calculate standard physicochemical descriptors for 2D molecular structure."""
    return {
        "MW": round(float(Descriptors.MolWt(mol)), 2), "LogP": round(float(Descriptors.MolLogP(mol)), 2),
        "HBD": int(Lipinski.NumHDonors(mol)), "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotBonds": int(Lipinski.NumRotatableBonds(mol)), "AromRings": int(Lipinski.NumAromaticRings(mol)),
        "TPSA": round(float(Descriptors.TPSA(mol)), 2),
    }


def predict(smiles: str, threshold: float = 6.0, run_rf: bool = True) -> Dict[str, Any]:
    """Run multi-target affinity prediction, adaptive conformal bounds, functional MoA, safety, and ADMET."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("Invalid SMILES")
    canon = Chem.MolToSmiles(mol, canonical=True)

    lookup = _load_db_lookup()
    in_db = canon in lookup
    ad_info = check_applicability_domain(canon)
    scaler = _load_scaler("precise")
    x = build_features(canon, scaler) if scaler is not None else None

    models_map = {"XGBoost": _load_xgb_models(), "RandomForest": _load_rf_models(), "LightGBM": _load_lgb_models()}
    stack_models = _load_stack_models()

    preds: Dict[str, Dict[str, float]] = {m: {} for m in ["XGBoost", "RandomForest", "LightGBM", "Stacked", "PyTorch", "MultiTask_Covariance"]}
    unc: Dict[str, Dict[str, float]] = {m: {} for m in preds}
    intervals: Dict[str, Dict[str, Any]] = {m: {} for m in preds}
    ki_values: Dict[str, Any] = {}

    for st in SUBTYPES:
        for m_name, m_dict in models_map.items():
            p, u, iv = _predict_one_subtype(m_dict, x, st)
            # Apply locally adaptive / scaffold-conditioned conformal scaling
            iv_adaptive = calibrate_adaptive_interval(p, iv["lower"], iv["upper"], ad_info.get("tanimoto_max", 0.0), in_domain=ad_info["in_domain"])
            preds[m_name][st], unc[m_name][st], intervals[m_name][st] = p, u, iv_adaptive

        # Stacked meta-learner blending
        valid_bases = [preds[m].get(st, 0.0) for m in ("XGBoost", "RandomForest", "LightGBM") if preds[m].get(st, 0.0) > 0]
        m_val = float(stack_models[st].predict(np.array([valid_bases]))[0]) if (st in stack_models and x is not None and len(valid_bases) == 3) else (float(np.mean(valid_bases)) if valid_bases else 0.0)
        preds["Stacked"][st], unc["Stacked"][st] = round(m_val, 3), 0.0
        intervals["Stacked"][st] = {"lower": round(m_val, 3), "upper": round(m_val, 3), "width": 0.0}

        # GNN evaluation
        gnn_val = _try_gnn_predict(canon, st)
        preds["PyTorch"][st], unc["PyTorch"][st], intervals["PyTorch"][st] = ((float(gnn_val), 0.0, {"lower": float(gnn_val), "upper": float(gnn_val), "width": 0.0}) if gnn_val is not None else _ZERO_RESULT)

    # Multi-task 7-TM covariance regularization & thermodynamic Ki (nM) conversion
    ref_preds = preds["XGBoost"]
    cov_preds = regularize_multitask_predictions(ref_preds)
    preds["MultiTask_Covariance"] = cov_preds

    for st in SUBTYPES:
        ki_nm = pchembl_to_ki_nm(ref_preds.get(st, 0.0))
        ki_low = pchembl_to_ki_nm(intervals["XGBoost"][st]["upper"])  # higher pChEMBL = lower Ki nM
        ki_high = pchembl_to_ki_nm(intervals["XGBoost"][st]["lower"])
        ki_values[st] = {"ki_nm": round(ki_nm, 2), "display": format_ki_display(ki_nm), "interval_display": f"[{format_ki_display(ki_low)} – {format_ki_display(ki_high)}]"}

    sel_spectrum = compute_selectivity_spectrum(ref_preds, canon, in_domain=ad_info["in_domain"])
    target_hits = [st for st, score in ref_preds.items() if score >= threshold and ad_info["in_domain"]]
    primary_target = sel_spectrum["best_target"]
    top_pchembl = float(ref_preds.get(primary_target, 0.0) or 0.0)

    # Advanced Translational Engines (Pillars 1 to 4)
    efficacy_profile = predict_functional_efficacy(canon, primary_target, top_pchembl, in_domain=ad_info["in_domain"])
    pocket_profile = analyze_pocket_interactions(canon, primary_target)
    safety_profile = evaluate_safety_profile(ref_preds, canon, in_domain=ad_info["in_domain"])
    cns_admet = evaluate_cns_admet(canon)
    db_exp = {st: lookup[canon].get(st) for st in SUBTYPES} if in_db and isinstance(lookup.get(canon), dict) else None

    return {
        "smiles": canon, "predictions": preds, "uncertainty": unc, "intervals": intervals,
        "ki_values": ki_values, "best_target": primary_target, "target_hits": target_hits,
        "selectivity_profile": sel_spectrum["pairwise_deltas"], "selectivity_spectrum": sel_spectrum,
        "applicability_domain": ad_info, "functional_efficacy": efficacy_profile,
        "pocket_interactions": pocket_profile, "safety_profile": safety_profile,
        "cns_admet": cns_admet, "descriptors": _descriptors(mol), "in_database": in_db,
        "db_value": db_exp, "source": "database" if in_db else "model",
    }


def batch_predict(smiles_list: List[str], threshold: float = 6.0) -> List[Dict[str, Any]]:
    """Execute prediction pipeline over a list of SMILES."""
    return [predict(s, threshold=threshold) for s in smiles_list if s.strip()]
