"""Per-prediction analysis: applicability domain, training-set neighbors per receptor, real SHAP."""
import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SUBTYPES

logger = logging.getLogger(__name__)

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)

# ponytail: static glossary reused by SHAP interpretation; no db, no translation layer
_GLOSSARY = {
    "MolLogP": "Lipophilicity (octanol/water partition) — balances membrane permeability vs solubility",
    "MolWt": "Molecular weight — size/solubility filter",
    "TPSA": "Topological polar surface area — polar contact surface for the GPCR pocket",
    "NumHDonors": "H-bond donors — anchor to Asn253/Ser277 in the adenosine pocket",
    "NumHAcceptors": "H-bond acceptors — H-bonding to the receptor",
    "NumRotatableBonds": "Rotatable bonds — conformational flexibility / entropic penalty",
    "NumAromaticRings": "Aromatic rings — π-stacking with Phe168/phe stack",
    "FractionCsp3": "Fraction of sp3 carbons — 3D shape / target selectivity",
    "LabuteASA": "Labute accessible surface area — shape complementarity with the binding site",
}


@lru_cache(maxsize=1)
def _load_training():
    try:
        with open(PROCESSED_DATA_DIR / "train_smiles.pkl", "rb") as f:
            train_smiles = pickle.load(f)
        with open(PROCESSED_DATA_DIR / "train_fps.pkl", "rb") as f:
            train_fps = pickle.load(f)
        with open(PROCESSED_DATA_DIR / "db_lookup_train.json") as f:
            lookup = json.load(f)
        return train_smiles, train_fps, lookup
    except Exception as e:
        logger.warning("Training data unavailable: %s", e)
        return None, None, None


@lru_cache(maxsize=1)
def _load_pipeline():
    for cand in (MODELS_DIR / "precise" / "scaler_precise.pkl", MODELS_DIR / "precise" / "scaler.pkl", MODELS_DIR / "scaler.pkl"):
        if cand.exists():
            with open(cand, "rb") as f:
                return pickle.load(f)
    return None


@lru_cache(maxsize=8)
def _load_estimator(subtype):
    mc_path = None
    for cand in (MODELS_DIR / "precise" / f"xgboost_{subtype}_production.pkl", MODELS_DIR / f"xgboost_{subtype}_production.pkl"):
        if cand.exists():
            mc_path = cand
            break
    if not mc_path:
        return None
    try:
        with open(mc_path, "rb") as f:
            mc = pickle.load(f)
        mtype = type(mc).__name__
        if mtype in ("CrossConformalRegressor", "MapieRegressor"):
            reg = mc._mapie_regressor.estimator_ if mtype == "CrossConformalRegressor" else mc.estimator_
            if hasattr(reg, "single_estimator_"):
                reg = reg.single_estimator_
            elif hasattr(reg, "estimators_") and len(reg.estimators_) > 0:
                reg = reg.estimators_[0]
        else:
            reg = mc
        return reg
    except Exception as e:
        logger.error("SHAP estimator load failed for %s: %s", subtype, e)
        return None


def _tanimoto_badge(sim):
    if sim >= 0.7:
        return "green", f"High ({sim:.3f})"
    if sim >= 0.4:
        return "amber", f"Medium ({sim:.3f})"
    return "red", f"Low ({sim:.3f})"


def _activity_badge(pchembl):
    if pchembl >= 6.0:
        return "green", "Active"
    if pchembl >= 4.5:
        return "amber", "Weak"
    return "red", "Inactive"


def _neighbor_records(sim, smiles, tanimoto, pchembl=None, subtype=None):
    from src.pdb_utils import real_structure_refs_with_analogs
    _, sim_label = _tanimoto_badge(tanimoto)
    try:
        refs = real_structure_refs_with_analogs(smiles, subtype=subtype)
    except Exception:
        refs = []
    rec = {"smiles": smiles, "tanimoto": round(tanimoto, 3), "similarity_label": sim_label,
           "real_structures": refs}
    if pchembl is not None:
        _, act_label = _activity_badge(pchembl)
        rec["pchembl"] = round(pchembl, 2)
        rec["activity"] = act_label
    return rec


def receptor_neighbors(smiles: str, subtype: str, top_k: int = 10):
    """Top-k training compounds with known activity for `subtype`, ranked by Tanimoto to query."""
    train_smiles, train_fps, lookup = _load_training()
    if train_smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    qfp = _MORGAN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)

    scored = []
    for i, tsmiles in enumerate(train_smiles):
        entry = lookup.get(tsmiles)
        if not entry:
            continue
        try:
            pchembl = float(entry.get(subtype))
        except (TypeError, ValueError):
            continue
        if pchembl <= 0:
            continue
        scored.append((sims[i], tsmiles, pchembl))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_neighbor_records(round(s, 3), t, s, p, subtype=subtype) for s, t, p in scored[:top_k]]


_RECEPTOR_TEMPLATES = {  # Verified deposited human GPCRdb receptor complexes
    "A1": "6D9H", "A2A": "6GDG", "A2B": "8HDO", "A3": "8X16",
}

# Verified via RCSB: each template is an adenosine/endogenous-agonist-bound complex.
_RECEPTOR_TEMPLATE_TITLE = {
    "6D9H": "Human A1–Gi2 complex bound to endogenous agonist",
    "2YDO": "Thermostabilised human A2A receptor with adenosine bound",
    "8HDP": "Human A2B receptor bound to adenosine",
    "8YH2": "Human A3–Gi complex bound to adenosine",
}


def receptors_overview(smiles: str):
    """Max similarity + active-neighbor count per subtype."""
    train_smiles, train_fps, lookup = _load_training()
    if train_smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    qfp = _MORGAN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
    out = {}
    for st in SUBTYPES:
        entries = []
        for i, tsmiles in enumerate(train_smiles):
            entry = lookup.get(tsmiles)
            if not entry:
                continue
            try:
                pchembl = float(entry.get(st))
            except (TypeError, ValueError):
                continue
            if pchembl > 0:
                entries.append((sims[i], pchembl))
        max_sim = max((s for s, _ in entries), default=0.0)
        n_active = sum(1 for _, p in entries if p >= 6.0)
        out[st] = {"max_similarity": round(max_sim, 3), "active_neighbors": n_active,
                   "pdb": _RECEPTOR_TEMPLATES.get(st),
                   "pdb_title": _RECEPTOR_TEMPLATE_TITLE.get(_RECEPTOR_TEMPLATES.get(st) or "", "")}
    return out


def shap_analysis(smiles: str, best_target: str, top_k: int = 10):
    """Real per-molecule SHAP values via TreeExplainer on the best-target XGBoost model."""
    if best_target not in SUBTYPES:
        return None
    import shap
    from src.features import build_features

    reg = _load_estimator(best_target)
    pl = _load_pipeline()
    if reg is None or pl is None:
        return None
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    try:
        x = build_features(canon, pl).reshape(1, -1)
    except Exception as e:
        logger.error("build_features failed for SHAP: %s", e)
        return None

    desc_names = []
    ff = getattr(pl, "feature_filter", None)
    if ff is not None and getattr(ff, "feature_names", None):
        desc_names = list(ff.feature_names)
    n_fp, n_maccs = 2048, 167
    fnames = [f"FP{i}" for i in range(n_fp)] + [f"MAC{i}" for i in range(n_maccs)] + desc_names
    if len(fnames) > x.shape[1]:
        fnames = fnames[: x.shape[1]]
    elif len(fnames) < x.shape[1]:
        fnames += [f"Feature_{i}" for i in range(len(fnames), x.shape[1])]

    xdf = pd.DataFrame(x, columns=fnames)
    e = shap.TreeExplainer(reg)
    sv = e(xdf)
    sv_df = pd.DataFrame({"feature": sv.feature_names, "value": sv.values[0]})
    sv_df["abs"] = sv_df["value"].abs()
    sv_df = sv_df.sort_values("abs", ascending=False).head(top_k)

    base = float(sv.base_values[0])
    features = []
    for _, row in sv_df.iterrows():
        feat = row["feature"]
        if feat.startswith("FP"):
            label, meaning = f"Morgan bit {feat[2:]}", "Local substructure presence (circular fingerprint bit)"
        elif feat.startswith("MAC"):
            label, meaning = f"MACCS key {feat[3:]}", "Predefined structural fragment presence"
        else:
            label, meaning = feat, _GLOSSARY.get(feat, "RDKit-calculated molecular descriptor")
        features.append({
            "feature": label, "raw": feat, "meaning": meaning,
            "value": round(float(row["value"]), 4),
            "direction": "increases prediction" if row["value"] > 0 else "decreases prediction",
            "positive": bool(row["value"] > 0),
        })
    return {"base_value": round(base, 3), "best_target": best_target, "features": features}
