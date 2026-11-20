"""Per-prediction analysis: training-set chemical neighbors and TreeSHAP feature attributions."""
import json
import logging
import pickle
from functools import lru_cache
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.config import MODELS_DIR, MORGAN_BITS, MORGAN_RADIUS, PROCESSED_DATA_DIR, SUBTYPES

logger = logging.getLogger(__name__)

_MORGAN = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

_GLOSSARY = {
    "MolLogP": "Lipophilicity (octanol/water partition): balances membrane permeability vs solubility",
    "MolWt": "Molecular weight: size and permeability filter",
    "TPSA": "Topological polar surface area: polar contact surface for the GPCR pocket",
    "NumHDonors": "H-bond donors: anchor to Asn6.55 in the adenosine pocket",
    "NumHAcceptors": "H-bond acceptors: H-bonding to receptor residues",
    "NumRotatableBonds": "Rotatable bonds: conformational flexibility and entropic penalty",
    "NumAromaticRings": "Aromatic rings: pi-stacking with ECL2 Phe168/Phe171",
    "FractionCSP3": "Fraction of sp3 carbons: 3D character and subtype selectivity",
    "LabuteASA": "Labute accessible surface area: shape complementarity with the binding pocket",
}


@lru_cache(maxsize=2)
def _load_training_cached(
    smiles_mtime: int,
    fps_mtime: int,
    lookup_mtime: int,
):
    with open(PROCESSED_DATA_DIR / "train_smiles.pkl", "rb") as f:
        train_smiles = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "train_fps.pkl", "rb") as f:
        train_fps = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "db_lookup_train.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
    return train_smiles, train_fps, lookup


def _load_training():
    smiles_p = PROCESSED_DATA_DIR / "train_smiles.pkl"
    fps_p = PROCESSED_DATA_DIR / "train_fps.pkl"
    lookup_p = PROCESSED_DATA_DIR / "db_lookup_train.json"
    if not (smiles_p.exists() and fps_p.exists() and lookup_p.exists()):
        return None, None, None
    try:
        return _load_training_cached(
            smiles_p.stat().st_mtime_ns,
            fps_p.stat().st_mtime_ns,
            lookup_p.stat().st_mtime_ns,
        )
    except Exception as exc:
        logger.warning("Training data unavailable: %s", exc)
        return None, None, None


@lru_cache(maxsize=8)
def _load_pipeline_cached(path_str: str, mtime_ns: int):
    return joblib.load(path_str)


def _load_pipeline(subtype: Optional[str] = None):
    if subtype is not None:
        st_cand = MODELS_DIR / "precise" / f"scaler_{subtype}.pkl"
        if st_cand.exists():
            try:
                return _load_pipeline_cached(str(st_cand), st_cand.stat().st_mtime_ns)
            except Exception as exc:
                logger.warning("Failed loading subtype scaler pipeline %s: %s", st_cand, exc)
    cand = MODELS_DIR / "precise" / "scaler_precise.pkl"
    if not cand.exists():
        return None
    try:
        return _load_pipeline_cached(str(cand), cand.stat().st_mtime_ns)
    except Exception as exc:
        logger.warning("Failed loading scaler pipeline %s: %s", cand, exc)
        return None


@lru_cache(maxsize=8)
def _load_estimator_cached(path_str: str, mtime_ns: int):
    mc = joblib.load(path_str)
    mtype = type(mc).__name__
    if mtype in ("CrossConformalRegressor", "MapieRegressor"):
        reg = mc._mapie_regressor.estimator_ if mtype == "CrossConformalRegressor" else mc.estimator_
        single_est = getattr(reg, "single_estimator_", None)
        if single_est is not None:
            return single_est
        estimators = getattr(reg, "estimators_", None)
        if estimators:
            return estimators[0]
        return reg
    return mc


def _load_estimator(subtype: str):
    mc_path = MODELS_DIR / "precise" / f"xgboost_{subtype}_production.pkl"
    if not mc_path.exists():
        return None
    try:
        return _load_estimator_cached(str(mc_path), mc_path.stat().st_mtime_ns)
    except Exception as exc:
        logger.error("SHAP estimator load failed for %s: %s", subtype, exc)
        return None


def _tanimoto_badge(sim: float) -> tuple[str, str]:
    if sim >= 0.7:
        return "green", f"High ({sim:.3f})"
    if sim >= 0.4:
        return "amber", f"Medium ({sim:.3f})"
    return "red", f"Low ({sim:.3f})"


def _activity_badge(pchembl: float) -> tuple[str, str]:
    if pchembl >= 6.0:
        return "green", "Active"
    if pchembl >= 4.5:
        return "amber", "Weak"
    return "red", "Inactive"


def _neighbor_record(smiles: str, tanimoto: float, pchembl: float) -> Dict[str, Any]:
    _, sim_label = _tanimoto_badge(tanimoto)
    _, act_label = _activity_badge(pchembl)
    return {
        "smiles": smiles,
        "tanimoto": round(float(tanimoto), 3),
        "similarity_label": sim_label,
        "pchembl": round(float(pchembl), 2),
        "activity": act_label,
    }


def receptor_neighbors(smiles: str, subtype: str, top_k: int = 10):
    """Return top-k training compounds with measured activity at `subtype`, ranked by Tanimoto."""
    if subtype not in SUBTYPES:
        return None
    train_smiles, train_fps, lookup = _load_training()
    if train_smiles is None or train_fps is None or lookup is None:
        return None
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None

    qfp = _MORGAN.GetFingerprint(mol)
    n_items = min(len(train_smiles), len(train_fps))
    valid_pairs = [(i, train_fps[i]) for i in range(n_items) if train_fps[i] is not None]
    if not valid_pairs:
        return []

    valid_idx = [i for i, _ in valid_pairs]
    valid_fps = [fp for _, fp in valid_pairs]
    sims = DataStructs.BulkTanimotoSimilarity(qfp, valid_fps)

    scored = []
    seen = set()
    for pos, sim in enumerate(sims):
        tsmiles = train_smiles[valid_idx[pos]]
        if tsmiles in seen:
            continue
        entry = lookup.get(tsmiles)
        if not entry:
            continue
        raw_val = entry.get(subtype)
        if raw_val is None:
            continue
        try:
            pchembl = float(raw_val)
        except (TypeError, ValueError):
            continue
        if pchembl <= 0:
            continue
        seen.add(tsmiles)
        scored.append((float(sim), tsmiles, pchembl))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_neighbor_record(t, s, p) for s, t, p in scored[:top_k]]


def shap_analysis(smiles: str, subtype: str, top_k: int = 10) -> Optional[Dict[str, Any]]:
    """Compute per-molecule TreeSHAP attributions on the conformal XGBoost base estimator."""
    if subtype not in SUBTYPES:
        return None
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None

    reg = _load_estimator(subtype)
    pl = _load_pipeline(subtype)
    if reg is None or pl is None:
        return None

    import shap
    from src.features import build_features

    canon = Chem.MolToSmiles(mol, canonical=True)
    try:
        x = build_features(canon, pl).reshape(1, -1)
    except Exception as exc:
        logger.error("Feature construction failed for SHAP (%s): %s", subtype, exc)
        return None

    desc_names = []
    ff = getattr(pl, "feature_filter", None)
    if ff is not None and getattr(ff, "feature_names", None):
        desc_names = list(ff.feature_names)

    fnames = [f"FP{i}" for i in range(MORGAN_BITS)] + [f"MAC{i}" for i in range(167)] + desc_names
    if len(fnames) > x.shape[1]:
        fnames = fnames[: x.shape[1]]
    elif len(fnames) < x.shape[1]:
        fnames += [f"Feature_{i}" for i in range(len(fnames), x.shape[1])]

    xdf = pd.DataFrame(x, columns=fnames)
    explainer = shap.TreeExplainer(reg)
    sv = explainer(xdf)
    sv_df = pd.DataFrame({"feature": sv.feature_names, "value": sv.values[0]})
    sv_df["abs"] = sv_df["value"].abs()
    sv_df = sv_df.sort_values("abs", ascending=False).head(top_k)

    base = float(np.asarray(sv.base_values).ravel()[0])
    features = []
    for _, row in sv_df.iterrows():
        feat = str(row["feature"])
        val = float(row["value"])
        if feat.startswith("FP"):
            label = f"Morgan bit {feat[2:]}"
            meaning = "Local substructure presence (circular fingerprint bit)"
        elif feat.startswith("MAC"):
            label = f"MACCS key {feat[3:]}"
            meaning = "Predefined structural fragment presence"
        else:
            label = feat
            meaning = _GLOSSARY.get(feat, "RDKit-calculated molecular descriptor")
        features.append(
            {
                "feature": label,
                "raw": feat,
                "meaning": meaning,
                "value": round(val, 4),
                "direction": "increases prediction" if val > 0 else "decreases prediction",
                "positive": bool(val > 0),
            }
        )

    return {
        "base_value": round(base, 3),
        "subtype": subtype,
        "best_target": subtype,
        "features": features,
    }
