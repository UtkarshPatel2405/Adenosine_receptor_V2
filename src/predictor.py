"""Master prediction API coordinating 3-model conformal consensus (XGBoost/LightGBM/RF), AD, and translational engines."""
import math
from typing import Any, Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from src.applicability_domain import check_applicability_domain
from src.chem_utils import canonicalize, check_pains, qed_profile
from src.config import SUBTYPES
from src.features import build_features
from src.models.adaptive_conformal import calibrate_adaptive_interval
from src.models.admet_engine import evaluate_cns_admet
from src.models.efficacy_engine import predict_functional_efficacy
from src.models.ensemble_engine import _ZERO_RESULT, _ensemble_predict, _predict_one_subtype
from src.models.interaction_engine import analyze_pocket_interactions
from src.models.ki_utils import format_ki_display, pchembl_to_ki_nm
from src.models.model_loader import (
    _load_consensus_weights,
    _load_db_lookup,
    _load_db_lookup_functional,
    _load_db_lookup_functional_train,
    _load_db_lookup_train,
    _load_lgb_models,
    _load_rf_models,
    _load_scaler,
    _load_xgb_models,
)
from src.models.safety_engine import evaluate_safety_profile
from src.models.selectivity_engine import compute_selectivity_spectrum


def _descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    """Calculate standard physicochemical descriptors for 2D molecular structure."""
    return {
        "MW": round(float(Descriptors.MolWt(mol)), 2),
        "LogP": round(float(Descriptors.MolLogP(mol)), 2),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
        "AromRings": int(Lipinski.NumAromaticRings(mol)),
        "TPSA": round(float(Descriptors.TPSA(mol)), 2),
    }


def _align_interval_with_pred(pred_val: float, iv: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee 3.0 <= lower <= pred_val <= upper <= 11.0 when pred_val lies in [3.0, 11.0]."""
    out = dict(iv)
    low = float(out["lower"])
    high = float(out["upper"])
    if low > high:
        low, high = high, low
    low = max(3.0, min(11.0, low))
    high = max(3.0, min(11.0, high))
    if 3.0 <= pred_val <= 11.0:
        low = min(low, round(math.floor(pred_val * 100.0) / 100.0, 2))
        high = max(high, round(math.ceil(pred_val * 100.0) / 100.0, 2))
    out["lower"] = round(max(3.0, low), 2)
    out["upper"] = round(min(11.0, high), 2)
    out["width"] = round(max(0.0, out["upper"] - out["lower"]), 2)
    return out


def _run_3model_suite(
    canon: str,
    ad_info: Dict[str, Any],
    folder: str = "precise",
) -> tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Dict[str, float]],
]:
    """Execute 3-model conformal prediction + validation-weighted consensus using independent per-subtype scalers."""
    mode = "functional" if folder == "functional" else "precise"
    xgb_models = _load_xgb_models(folder)
    lgb_models = _load_lgb_models(folder)
    rf_models = _load_rf_models(folder)
    weights_map = _load_consensus_weights(folder)
    models_map = {"XGBoost": xgb_models, "LightGBM": lgb_models, "RandomForest": rf_models}

    preds: Dict[str, Dict[str, float]] = {
        m: {} for m in ("XGBoost", "LightGBM", "RandomForest", "Consensus")
    }
    unc: Dict[str, Dict[str, float]] = {m: {} for m in preds}
    intervals: Dict[str, Dict[str, Any]] = {m: {} for m in preds}
    potency_values: Dict[str, Any] = {}

    for st in SUBTYPES:
        try:
            st_scaler = _load_scaler(mode, subtype=st)
            x_st = build_features(canon, st_scaler) if st_scaler is not None else None
        except Exception:
            x_st = None

        raw_ivs: Dict[str, Dict[str, float]] = {}
        available_models: List[str] = []

        for m_name, m_dict in models_map.items():
            if st in m_dict and x_st is not None:
                p, u, iv_raw = _predict_one_subtype(m_dict, x_st, st)
                iv_cal = calibrate_adaptive_interval(
                    p,
                    iv_raw["lower"],
                    iv_raw["upper"],
                    ad_info.get("tanimoto_max", 0.0),
                    in_domain=ad_info["in_domain"],
                )
                p_rounded = round(float(p), 3)
                preds[m_name][st] = p_rounded
                unc[m_name][st] = round(float(u), 4)
                intervals[m_name][st] = _align_interval_with_pred(p_rounded, iv_cal)
                raw_ivs[m_name] = iv_raw
                available_models.append(m_name)
            else:
                p, u, iv_zero = _ZERO_RESULT
                preds[m_name][st] = p
                unc[m_name][st] = u
                intervals[m_name][st] = dict(iv_zero)

        if available_models:
            w_st = weights_map.get(st, {"XGBoost": 1 / 3, "LightGBM": 1 / 3, "RandomForest": 1 / 3})
            w_sum = sum(float(w_st.get(m, 1 / 3)) for m in available_models)
            if w_sum <= 0:
                c_pred = float(np.mean([preds[m][st] for m in available_models]))
            else:
                c_pred = float(
                    sum(float(w_st.get(m, 1 / 3)) * preds[m][st] for m in available_models) / w_sum
                )

            conformal_sources = [m for m in ("XGBoost", "LightGBM", "RandomForest") if m in raw_ivs]
            iv_sources = conformal_sources if conformal_sources else available_models
            iv_w_sum = sum(float(w_st.get(m, 0.5)) for m in iv_sources)
            if iv_w_sum > 0:
                c_unc = float(sum(float(w_st.get(m, 0.5)) * unc[m][st] for m in iv_sources) / iv_w_sum)
                c_low = float(sum(float(w_st.get(m, 0.5)) * raw_ivs[m]["lower"] for m in iv_sources) / iv_w_sum)
                c_high = float(sum(float(w_st.get(m, 0.5)) * raw_ivs[m]["upper"] for m in iv_sources) / iv_w_sum)
            else:
                c_unc = float(np.mean([unc[m][st] for m in iv_sources]))
                c_low = float(np.mean([raw_ivs[m]["lower"] for m in iv_sources]))
                c_high = float(np.mean([raw_ivs[m]["upper"] for m in iv_sources]))

            c_iv = calibrate_adaptive_interval(
                c_pred,
                c_low,
                c_high,
                ad_info.get("tanimoto_max", 0.0),
                in_domain=ad_info["in_domain"],
            )
            c_rounded = round(c_pred, 3)
            preds["Consensus"][st] = c_rounded
            unc["Consensus"][st] = round(c_unc, 4)
            intervals["Consensus"][st] = _align_interval_with_pred(c_rounded, c_iv)
        else:
            p, u, iv_zero = _ZERO_RESULT
            preds["Consensus"][st] = p
            unc["Consensus"][st] = u
            intervals["Consensus"][st] = dict(iv_zero)

    ref_preds = preds["Consensus"]
    ref_intervals = intervals["Consensus"]
    for st in SUBTYPES:
        ki_nm = pchembl_to_ki_nm(ref_preds.get(st, 0.0))
        ki_low = pchembl_to_ki_nm(ref_intervals[st]["upper"])
        ki_high = pchembl_to_ki_nm(ref_intervals[st]["lower"])
        potency_values[st] = {
            "ki_nm": round(ki_nm, 2),
            "display": format_ki_display(ki_nm),
            "interval_display": f"[{format_ki_display(ki_low)} - {format_ki_display(ki_high)}]",
        }
    return preds, unc, intervals, potency_values, weights_map


def predict(smiles: str, threshold: float = 6.0) -> Dict[str, Any]:
    """Run conformal XGBoost + LightGBM + RandomForest validation-weighted consensus (Model 1: pKi, Model 2: pIC50/pEC50)."""
    canon = canonicalize(smiles) if isinstance(smiles, str) else None
    if canon is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        raise ValueError("Invalid SMILES")

    train_lookup = _load_db_lookup_train()
    full_lookup = _load_db_lookup()
    if canon in train_lookup:
        db_status = "training"
        raw_entry = train_lookup.get(canon)
    elif canon in full_lookup:
        db_status = "test"
        raw_entry = full_lookup.get(canon)
    else:
        db_status = "de_novo"
        raw_entry = None

    in_db = db_status != "de_novo"
    db_exp = (
        {st: raw_entry.get(st) for st in SUBTYPES}
        if in_db and isinstance(raw_entry, dict)
        else None
    )

    func_train_lookup = _load_db_lookup_functional_train()
    func_full_lookup = _load_db_lookup_functional()
    if canon in func_train_lookup:
        func_db_status = "training"
        func_raw_entry = func_train_lookup.get(canon)
    elif canon in func_full_lookup:
        func_db_status = "test"
        func_raw_entry = func_full_lookup.get(canon)
    else:
        func_db_status = "de_novo"
        func_raw_entry = None

    in_func_db = func_db_status != "de_novo"
    func_db_exp = (
        {st: func_raw_entry.get(st) for st in SUBTYPES}
        if in_func_db and isinstance(func_raw_entry, dict)
        else None
    )

    ad_info = check_applicability_domain(canon)
    preds, unc, intervals, ki_values, consensus_weights = _run_3model_suite(canon, ad_info, folder="precise")
    func_preds, func_unc, func_intervals, func_potency_values, func_weights = _run_3model_suite(
        canon, ad_info, folder="functional"
    )

    ref_preds = preds["Consensus"]
    ref_intervals = intervals["Consensus"]

    sel_spectrum = compute_selectivity_spectrum(ref_preds, canon, in_domain=ad_info["in_domain"])
    target_hits = [st for st, score in ref_preds.items() if score >= threshold and ad_info["in_domain"]]
    highest_target = sel_spectrum["best_target"]
    top_pchembl = float(ref_preds.get(highest_target, 0.0) or 0.0)
    if target_hits:
        primary_target = highest_target
    else:
        primary_target = "None (Inactive across all subtypes)"

    efficacy_profile = predict_functional_efficacy(
        canon,
        highest_target if target_hits else "None",
        top_pchembl,
        in_domain=ad_info["in_domain"],
        threshold=threshold,
    )
    pocket_profile = analyze_pocket_interactions(canon, highest_target if highest_target in SUBTYPES else "A2A")
    safety_profile = evaluate_safety_profile(ref_preds, canon, in_domain=ad_info["in_domain"])
    cns_admet = evaluate_cns_admet(canon)

    return {
        "smiles": canon,
        "predictions": preds,
        "uncertainty": unc,
        "intervals": intervals,
        "ki_values": ki_values,
        "best_target": primary_target,
        "target_hits": target_hits,
        "selectivity_profile": sel_spectrum["pairwise_deltas"],
        "selectivity_spectrum": sel_spectrum,
        "applicability_domain": ad_info,
        "functional_efficacy": efficacy_profile,
        "pocket_interactions": pocket_profile,
        "safety_profile": safety_profile,
        "cns_admet": cns_admet,
        "descriptors": _descriptors(mol),
        "qed": qed_profile(canon),
        "pains": check_pains(canon),
        "db_status": db_status,
        "in_database": in_db,
        "db_value": db_exp,
        "source": "database" if in_db else "model",
        "consensus_weights": consensus_weights,
        "functional_activity": {
            "assay_type": "pIC50 / pEC50 (Functional & Displacement Potency)",
            "predictions": func_preds,
            "uncertainty": func_unc,
            "intervals": func_intervals,
            "potency_values": func_potency_values,
            "consensus_weights": func_weights,
            "in_database": in_func_db,
            "db_status": func_db_status,
            "db_value": func_db_exp,
        },
    }


def batch_predict(smiles_list: List[str], threshold: float = 6.0) -> List[Dict[str, Any]]:
    """Execute prediction pipeline over a list of SMILES."""
    return [predict(s, threshold=threshold) for s in smiles_list if s.strip()]
