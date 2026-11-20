"""Production training of 5-fold MAPIE CrossConformal XGBoost, LightGBM, and compact RandomForest models."""
import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Any, Dict

import joblib
import lightgbm as lgb
import mapie
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from mapie.regression import CrossConformalRegressor
from rdkit import __version__ as rdkit_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from src.config import (
    LOG_LEVEL,
    MAPIE_CONFIDENCE,
    MAPIE_CV_FOLDS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RUN_ID,
    RUN_TIMESTAMP,
    SCAFFOLD_SPLIT_SEED,
    SCAFFOLD_TEST_SIZE,
    SUBTYPES,
)
from src.data_loader import load_and_clean
from src.features import build_feature_matrix
from src.scaffold_split import split_smiles_globally

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "A1": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 4,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "gamma": 0.1,
    },
    "A2A": {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 4,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "gamma": 0.1,
    },
    "A2B": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 4,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "gamma": 0.1,
    },
    "A3": {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 4,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "gamma": 0.1,
    },
}

LGBM_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "A1": {
        "n_estimators": 500,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "min_child_samples": 15,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
    "A2A": {
        "n_estimators": 600,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "min_child_samples": 15,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
    "A2B": {
        "n_estimators": 500,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "min_child_samples": 15,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
    "A3": {
        "n_estimators": 600,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "min_child_samples": 15,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
}


def train_conformal_model(base_model, X_train: np.ndarray, y_train: np.ndarray, cv: int = MAPIE_CV_FOLDS):
    """Wrap a scikit-learn compatible regressor in MAPIE CrossConformalRegressor (Jackknife+)."""
    conformal = CrossConformalRegressor(
        estimator=base_model,
        cv=cv,
        confidence_level=MAPIE_CONFIDENCE,
        method="plus",
        n_jobs=1,
    )
    conformal.fit_conformalize(X_train, y_train)
    logger.info("Conformal model trained: cv=%d, confidence=%.2f", cv, MAPIE_CONFIDENCE)
    return conformal


def _build_train_lookup(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for smi, sub, val in zip(
        train_df["smiles"].values,
        train_df["target_subtype"].values,
        train_df["pchembl_value"].values,
    ):
        lookup.setdefault(smi, {})[sub] = float(val)
    return lookup


def _compute_validation_weights(
    base_xgb,
    base_lgb,
    rf_model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_folds: int = MAPIE_CV_FOLDS,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Compute validation-weighted consensus weights from 5-fold out-of-fold RMSE (and RF OOB RMSE)."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_xgb = cross_val_predict(base_xgb, X_tr, y_tr, cv=kf, n_jobs=1)
    oof_lgb = cross_val_predict(base_lgb, X_tr, y_tr, cv=kf, n_jobs=1)
    rmse_xgb = float(np.sqrt(mean_squared_error(y_tr, oof_xgb)))
    rmse_lgb = float(np.sqrt(mean_squared_error(y_tr, oof_lgb)))
    if hasattr(rf_model, "oob_prediction_") and rf_model.oob_prediction_ is not None:
        rmse_rf = float(np.sqrt(mean_squared_error(y_tr, rf_model.oob_prediction_)))
    else:
        oof_rf = cross_val_predict(rf_model, X_tr, y_tr, cv=kf, n_jobs=1)
        rmse_rf = float(np.sqrt(mean_squared_error(y_tr, oof_rf)))

    inv_scores = {
        "XGBoost": 1.0 / max(rmse_xgb, 1e-3) ** 4,
        "LightGBM": 1.0 / max(rmse_lgb, 1e-3) ** 4,
        "RandomForest": 1.0 / max(rmse_rf, 1e-3) ** 4,
    }
    total_inv = sum(inv_scores.values())
    weights = {k: round(v / total_inv, 4) for k, v in inv_scores.items()}
    rmses = {
        "XGBoost": round(rmse_xgb, 4),
        "LightGBM": round(rmse_lgb, 4),
        "RandomForest": round(rmse_rf, 4),
    }
    return weights, rmses


def retrain_functional_models(
    data_path: str = "data/raw",
    train_smiles: set[str] | None = None,
    test_smiles: set[str] | None = None,
) -> Dict[str, Any]:
    """Train Use Case A: Functional Activity (pIC50 / pEC50) 3-model ensemble with 4 per-subtype scalers."""
    logger.info("Starting Use Case A: Functional Activity (pIC50/pEC50) 3-model training...")
    df_func, _ = load_and_clean(
        data_path,
        save_lookup_path=str(PROCESSED_DATA_DIR / "db_lookup_functional.json"),
        mode="precise",
        include_decoys=False,
        save_subtype_csvs=True,
        assay_mode="functional",
    )
    df_func = df_func.rename(columns={"canonical_smiles": "smiles"})

    if train_smiles is None or test_smiles is None:
        split_path = PROCESSED_DATA_DIR / "global_split.json"
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                sp = json.load(f)
            train_smiles = set(sp["train"])
            test_smiles = set(sp["test"])
        else:
            tr_s, te_s = split_smiles_globally(
                sorted(df_func["smiles"].unique()),
                test_size=SCAFFOLD_TEST_SIZE,
                random_state=SCAFFOLD_SPLIT_SEED,
            )
            train_smiles, test_smiles = set(tr_s), set(te_s)

    unseen_func = sorted(set(df_func["smiles"].unique()) - (train_smiles | test_smiles))
    if unseen_func:
        tr_u, te_u = split_smiles_globally(
            unseen_func, test_size=SCAFFOLD_TEST_SIZE, random_state=SCAFFOLD_SPLIT_SEED
        )
        train_smiles = set(train_smiles) | set(tr_u)
        test_smiles = set(test_smiles) | set(te_u)

    train_df = df_func[df_func["smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df_func[df_func["smiles"].isin(test_smiles)].reset_index(drop=True)

    train_lookup_func = _build_train_lookup(train_df)
    with open(PROCESSED_DATA_DIR / "db_lookup_functional_train.json", "w", encoding="utf-8") as f:
        json.dump(train_lookup_func, f, indent=2, sort_keys=True)

    func_dir = MODELS_DIR / "functional"
    func_dir.mkdir(parents=True, exist_ok=True)

    weights_func: Dict[str, Dict[str, float]] = {}
    summary_func: Dict[str, Any] = {}

    for st in SUBTYPES:
        tr_sub = train_df[train_df["target_subtype"] == st].reset_index(drop=True)
        te_sub = test_df[test_df["target_subtype"] == st].reset_index(drop=True)
        if len(tr_sub) < 30:
            logger.warning("Skipping functional %s: insufficient training rows (%d)", st, len(tr_sub))
            continue

        st_scaler_path = func_dir / f"scaler_{st}.pkl"
        X_tr, X_te, st_pipeline = build_feature_matrix(
            tr_sub,
            te_sub,
            smiles_col="smiles",
            save_to_disk=False,
            scaler_path=st_scaler_path,
        )
        joblib.dump(st_pipeline, st_scaler_path, compress=3)

        y_tr = tr_sub["pchembl_value"].to_numpy(dtype=float)
        y_te = te_sub["pchembl_value"].to_numpy(dtype=float)

        xgb_params = dict(DEFAULT_PARAMS[st])
        xgb_params.update({"tree_method": "hist", "n_jobs": 1, "random_state": 42})
        base_xgb = xgb.XGBRegressor(**xgb_params)
        xgb_conf = train_conformal_model(base_xgb, X_tr, y_tr, cv=MAPIE_CV_FOLDS)

        lgb_params = dict(LGBM_DEFAULT_PARAMS[st])
        lgb_params.update({"verbose": -1, "n_jobs": 1, "random_state": 42})
        base_lgb = lgb.LGBMRegressor(**lgb_params)
        lgb_conf = train_conformal_model(base_lgb, X_tr, y_tr, cv=MAPIE_CV_FOLDS)

        rf_model = RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        rf_conf = train_conformal_model(rf_model, X_tr, y_tr, cv=MAPIE_CV_FOLDS)

        w_st, rmses_st = _compute_validation_weights(base_xgb, base_lgb, rf_model, X_tr, y_tr)
        weights_func[st] = w_st

        joblib.dump(xgb_conf, func_dir / f"xgboost_{st}_production.pkl", compress=3)
        joblib.dump(lgb_conf, func_dir / f"lgb_{st}_production.pkl", compress=3)
        joblib.dump(rf_conf, func_dir / f"rf_{st}_production.pkl", compress=3)

        if len(y_te) > 0:
            p_xgb = np.asarray(xgb_conf.predict(X_te), dtype=float).ravel()
            p_lgb = np.asarray(lgb_conf.predict(X_te), dtype=float).ravel()
            p_rf = np.asarray(rf_conf.predict(X_te), dtype=float).ravel()
            p_cons = w_st["XGBoost"] * p_xgb + w_st["LightGBM"] * p_lgb + w_st["RandomForest"] * p_rf
            r2_c = float(r2_score(y_te, p_cons))
            mae_c = float(mean_absolute_error(y_te, p_cons))
        else:
            r2_c = mae_c = 0.0

        summary_func[st] = {
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "consensus_r2": round(r2_c, 4),
            "consensus_mae": round(mae_c, 4),
            "consensus_weights": w_st,
            "oof_rmse": rmses_st,
        }
        logger.info("Functional %s trained: N_train=%d, N_test=%d, Consensus R2=%.3f, weights=%s", st, len(y_tr), len(y_te), r2_c, w_st)

    with open(func_dir / "consensus_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_func, f, indent=2)
    return summary_func


def retrain_production_models(data_path: str = "data/raw") -> Dict[str, Any]:
    """Train conformal XGBoost + LightGBM and compact RandomForest models with 4 independent per-subtype scalers."""
    logger.info("Starting production 3-model training (XGBoost + LightGBM + RandomForest) with 4 independent subtype scalers...")

    df, _ = load_and_clean(
        data_path,
        save_lookup_path=str(PROCESSED_DATA_DIR / "db_lookup.json"),
        mode="precise",
        include_decoys=False,
        save_subtype_csvs=True,
        assay_mode="binding",
    )
    assay_filter = df.attrs.get("assay_filter", "Equilibrium binding Ki and Kd only (pKi)")
    n_dropped_conflict = int(df.attrs.get("n_replicates_dropped_conflict", 0))
    raw_sources = sorted(df.attrs.get("raw_sources_breakdown", {}).keys())
    dataset_source = " + ".join(raw_sources) if raw_sources else "GPCRdb + ChEMBL_Parents + ChEMBL_API"

    df = df.rename(columns={"canonical_smiles": "smiles"})

    unique_smiles = sorted(df["smiles"].unique())
    train_smiles, test_smiles = split_smiles_globally(
        unique_smiles,
        test_size=SCAFFOLD_TEST_SIZE,
        random_state=SCAFFOLD_SPLIT_SEED,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    split_path = PROCESSED_DATA_DIR / "global_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": sorted(train_smiles), "test": sorted(test_smiles)}, f, indent=2)
    logger.info("Saved global scaffold split to %s", split_path)

    train_df = df[df["smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["smiles"].isin(test_smiles)].reset_index(drop=True)

    train_lookup = _build_train_lookup(train_df)
    train_lookup_path = PROCESSED_DATA_DIR / "db_lookup_train.json"
    with open(train_lookup_path, "w", encoding="utf-8") as f:
        json.dump(train_lookup, f, indent=2, sort_keys=True)
    logger.info("Saved train-only lookup (%d molecules) to %s", len(train_lookup), train_lookup_path)

    models_precise_dir = MODELS_DIR / "precise"
    models_precise_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_precise_dir / "scaler_precise.pkl"

    X_train_glob, _, _ = build_feature_matrix(
        train_df,
        test_df,
        smiles_col="smiles",
        save_to_disk=True,
        scaler_path=scaler_path,
    )

    per_subtype_summary: Dict[str, Any] = {}
    consensus_weights_map: Dict[str, Dict[str, float]] = {}

    for st in SUBTYPES:
        logger.info("Training independent scaler and conformal models for subtype %s...", st)
        tr_sub = train_df[train_df["target_subtype"] == st].reset_index(drop=True)
        te_sub = test_df[test_df["target_subtype"] == st].reset_index(drop=True)

        if len(tr_sub) < 50:
            logger.warning("Skipping %s: insufficient training samples (%d).", st, len(tr_sub))
            continue

        st_scaler_path = models_precise_dir / f"scaler_{st}.pkl"
        X_tr, X_te, st_pipeline = build_feature_matrix(
            tr_sub,
            te_sub,
            smiles_col="smiles",
            save_to_disk=False,
            scaler_path=st_scaler_path,
        )
        joblib.dump(st_pipeline, st_scaler_path, compress=3)
        logger.info("Saved independent %s scaler (%d features) to %s", st, X_tr.shape[1], st_scaler_path)

        y_tr = tr_sub["pchembl_value"].to_numpy(dtype=float)
        y_te = te_sub["pchembl_value"].to_numpy(dtype=float)

        xgb_params = dict(DEFAULT_PARAMS[st])
        xgb_params.update({"tree_method": "hist", "n_jobs": 1, "random_state": 42})
        base_xgb = xgb.XGBRegressor(**xgb_params)
        xgb_conformal = train_conformal_model(base_xgb, X_tr, y_tr, cv=MAPIE_CV_FOLDS)

        lgb_params = dict(LGBM_DEFAULT_PARAMS[st])
        lgb_params.update({"verbose": -1, "n_jobs": 1, "random_state": 42})
        base_lgb = lgb.LGBMRegressor(**lgb_params)
        lgb_conformal = train_conformal_model(base_lgb, X_tr, y_tr, cv=MAPIE_CV_FOLDS)

        rf_base = RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        rf_conformal = train_conformal_model(rf_base, X_tr, y_tr, cv=MAPIE_CV_FOLDS)

        w_st, rmses_st = _compute_validation_weights(base_xgb, base_lgb, rf_base, X_tr, y_tr)
        consensus_weights_map[st] = w_st

        preds_te_xgb = np.asarray(xgb_conformal.predict(X_te), dtype=float).ravel()
        preds_te_lgb = np.asarray(lgb_conformal.predict(X_te), dtype=float).ravel()
        preds_te_rf = np.asarray(rf_conformal.predict(X_te), dtype=float).ravel()
        preds_te_cons = (
            w_st["XGBoost"] * preds_te_xgb
            + w_st["LightGBM"] * preds_te_lgb
            + w_st["RandomForest"] * preds_te_rf
        )

        r2_xgb = float(r2_score(y_te, preds_te_xgb)) if len(y_te) else 0.0
        r2_lgb = float(r2_score(y_te, preds_te_lgb)) if len(y_te) else 0.0
        r2_rf = float(r2_score(y_te, preds_te_rf)) if len(y_te) else 0.0
        r2_cons = float(r2_score(y_te, preds_te_cons)) if len(y_te) else 0.0
        mae_cons = float(mean_absolute_error(y_te, preds_te_cons)) if len(y_te) else 0.0

        xgb_path = models_precise_dir / f"xgboost_{st}_production.pkl"
        lgb_path = models_precise_dir / f"lgb_{st}_production.pkl"
        rf_path = models_precise_dir / f"rf_{st}_production.pkl"
        joblib.dump(xgb_conformal, xgb_path, compress=3)
        joblib.dump(lgb_conformal, lgb_path, compress=3)
        joblib.dump(rf_conformal, rf_path, compress=3)

        xgb_mb = xgb_path.stat().st_size / 1e6
        lgb_mb = lgb_path.stat().st_size / 1e6
        rf_mb = rf_path.stat().st_size / 1e6
        logger.info(
            "%s test metrics: XGB R2=%.3f (%.2f MB), LGB R2=%.3f (%.2f MB), RF R2=%.3f (%.2f MB), Weighted Consensus R2=%.3f, MAE=%.3f, weights=%s",
            st,
            r2_xgb,
            xgb_mb,
            r2_lgb,
            lgb_mb,
            r2_rf,
            rf_mb,
            r2_cons,
            mae_cons,
            w_st,
        )

        per_subtype_summary[st] = {
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "n_features": int(X_tr.shape[1]),
            "scaler_file": f"scaler_{st}.pkl",
            "xgboost_r2": round(r2_xgb, 5),
            "lgbm_r2": round(r2_lgb, 5),
            "rf_r2": round(r2_rf, 5),
            "consensus_r2": round(r2_cons, 5),
            "consensus_mae": round(mae_cons, 5),
            "consensus_weights": w_st,
            "oof_rmse": rmses_st,
            "xgb_size_mb": round(xgb_mb, 3),
            "lgb_size_mb": round(lgb_mb, 3),
            "rf_size_mb": round(rf_mb, 3),
            "conformal_wrapped": type(xgb_conformal).__name__,
        }

    with open(models_precise_dir / "consensus_weights.json", "w", encoding="utf-8") as f:
        json.dump(consensus_weights_map, f, indent=2)

    functional_summary = retrain_functional_models(
        data_path=data_path,
        train_smiles=set(train_smiles),
        test_smiles=set(test_smiles),
    )

    summary = {
        "run_id": RUN_ID,
        "trained_at": RUN_TIMESTAMP,
        "assay_filter": assay_filter,
        "dataset_source": dataset_source,
        "n_replicates_dropped_conflict": n_dropped_conflict,
        "n_records_total": int(len(df)),
        "n_unique_molecules": int(len(unique_smiles)),
        "n_train_molecules": int(len(train_smiles)),
        "n_test_molecules": int(len(test_smiles)),
        "n_features": int(X_train_glob.shape[1]),
        "independent_subtype_scalers": [f"scaler_{st}.pkl" for st in SUBTYPES],
        "consensus_weights": consensus_weights_map,
        "functional_models_summary": functional_summary,
        "subtypes": list(SUBTYPES),
        "models": ["XGBoost", "LightGBM", "RandomForest", "Consensus"],
        "package_versions": {
            "python": platform.python_version(),
            "rdkit": rdkit_version,
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "mapie": mapie.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "per_subtype": per_subtype_summary,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved training summary to %s [run_id=%s]", summary_path, RUN_ID)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train Conformal XGBoost + LightGBM Production Models")
    parser.add_argument("--data-dir", default="data/raw", help="Raw bioactivity data directory")
    args = parser.parse_args()
    retrain_production_models(data_path=args.data_dir)
