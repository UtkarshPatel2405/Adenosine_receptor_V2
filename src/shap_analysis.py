"""Global TreeSHAP explainability using saved MAPIE conformal XGBoost base estimators."""
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import shap

from src.config import (
    LOG_LEVEL,
    MODELS_DIR,
    MORGAN_BITS,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RUN_ID,
    SUBTYPES,
)
from src.data_loader import load_and_clean
from src.features import build_features

logger = logging.getLogger(__name__)


def _extract_xgb_estimator(model: Any):
    model_type = type(model).__name__
    if model_type in ("CrossConformalRegressor", "MapieRegressor"):
        reg = (
            model._mapie_regressor.estimator_
            if model_type == "CrossConformalRegressor"
            else model.estimator_
        )
        single_est = getattr(reg, "single_estimator_", None)
        if single_est is not None:
            return single_est
        estimators = getattr(reg, "estimators_", None)
        if estimators:
            return estimators[0]
        return reg
    if isinstance(model, list) and model:
        return model[0]
    return model


def _build_feature_names(pipeline: Any, n_cols: int) -> list[str]:
    desc_names = []
    ff = getattr(pipeline, "feature_filter", None)
    if ff is not None and getattr(ff, "feature_names", None):
        desc_names = list(ff.feature_names)
    feature_names = (
        [f"Morgan_FP_{i}" for i in range(MORGAN_BITS)]
        + [f"MACCS_{i}" for i in range(167)]
        + desc_names
    )
    if len(feature_names) > n_cols:
        return feature_names[:n_cols]
    if len(feature_names) < n_cols:
        feature_names.extend(f"Feature_{i}" for i in range(len(feature_names), n_cols))
    return feature_names


def _load_test_context(data_dir: str = "data/raw"):
    split_path = PROCESSED_DATA_DIR / "global_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Global split missing at {split_path}; run retrain_production first.")
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    scaler_path = MODELS_DIR / "precise" / "scaler_precise.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Production scaler missing at {scaler_path}.")
    pipeline = joblib.load(scaler_path)

    df, _ = load_and_clean(data_dir, save_lookup_path=None, mode="precise", include_decoys=False)
    test_smiles_set = set(splits["test"])
    test_df_full = df[df["canonical_smiles"].isin(test_smiles_set)].reset_index(drop=True)

    return test_df_full, None, pipeline


def run_shap_analysis(
    subtype: str,
    data_dir: str = "data/raw",
    top_k: int = 15,
    _context: Optional[tuple] = None,
) -> Optional[Dict[str, Any]]:
    """Compute mean |SHAP| feature importance on the held-out test set for `subtype`."""
    model_path = MODELS_DIR / "precise" / f"xgboost_{subtype}_production.pkl"
    if not model_path.exists():
        logger.error("Model missing for %s at %s.", subtype, model_path)
        return None

    model = joblib.load(model_path)
    estimator = _extract_xgb_estimator(model)

    test_df_full, X_test_glob, fallback_pipeline = _context if _context is not None else _load_test_context(data_dir)
    subtype_mask = (test_df_full["target_subtype"] == subtype).values
    if not np.any(subtype_mask):
        logger.warning("No held-out test records for %s.", subtype)
        return None

    st_scaler_path = MODELS_DIR / "precise" / f"scaler_{subtype}.pkl"
    if st_scaler_path.exists():
        pipeline = joblib.load(st_scaler_path)
        sub_smiles = test_df_full.loc[subtype_mask, "canonical_smiles"].tolist()
        X_test = np.vstack([build_features(s, pipeline) for s in sub_smiles]).astype(np.float32)
    elif X_test_glob is not None:
        pipeline = fallback_pipeline
        X_test = X_test_glob[subtype_mask]
    else:
        pipeline = fallback_pipeline
        sub_smiles = test_df_full.loc[subtype_mask, "canonical_smiles"].tolist()
        X_test = np.vstack([build_features(s, pipeline) for s in sub_smiles]).astype(np.float32)

    feature_names = _build_feature_names(pipeline, X_test.shape[1])

    explainer = shap.TreeExplainer(estimator)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    mean_abs = np.abs(np.asarray(shap_vals)).mean(axis=0)
    sorted_indices = np.argsort(mean_abs)[::-1][:top_k]

    top_features = [
        {"feature": feature_names[int(idx)], "mean_abs_shap": round(float(mean_abs[int(idx)]), 6)}
        for idx in sorted_indices
    ]

    logger.info("Computed TreeSHAP for %s (n_eval=%d, top=%s)", subtype, len(X_test), top_features[0]["feature"])
    return {
        "n_eval": int(len(X_test)),
        "top_features": top_features,
    }


def run_all_subtypes(data_dir: str = "data/raw", top_k: int = 15) -> Dict[str, Any]:
    """Compute global TreeSHAP summaries across all subtypes and write outputs/shap/shap_summary.json."""
    context = _load_test_context(data_dir)
    per_subtype: Dict[str, Any] = {}
    for st in SUBTYPES:
        res = run_shap_analysis(subtype=st, data_dir=data_dir, top_k=top_k, _context=context)
        if res is not None:
            per_subtype[st] = res

    summary = {
        "run_id": RUN_ID,
        "method": "TreeSHAP (XGBoost base estimator inside MAPIE CrossConformalRegressor)",
        "per_subtype": per_subtype,
    }

    out_dir = OUTPUTS_DIR / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved global TreeSHAP summary to %s", out_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Global TreeSHAP Attribution Summary")
    parser.add_argument("--subtype", choices=SUBTYPES, default=None, help="Single subtype to explain")
    parser.add_argument("--all", action="store_true", help="Run TreeSHAP across all subtypes")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--top-k", type=int, default=15, help="Number of top features to record")
    args = parser.parse_args()

    if args.subtype and not args.all:
        res = run_shap_analysis(subtype=args.subtype, data_dir=args.data_dir, top_k=args.top_k)
        if res is not None:
            out_dir = OUTPUTS_DIR / "shap"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.subtype}_shap_summary.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"run_id": RUN_ID, "subtype": args.subtype, **res}, f, indent=2)
    else:
        run_all_subtypes(data_dir=args.data_dir, top_k=args.top_k)
