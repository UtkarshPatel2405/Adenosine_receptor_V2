import json
import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


class AverageEnsemble:
    """Equal-weight average of base model predictions. Pickle-friendly."""
    def predict(self, X):
        return np.mean(X, axis=1)

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.scaffold_split import split_smiles_globally

from src.config import (
    SUBTYPES, MODELS_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR,
    SCAFFOLD_SPLIT_SEED, SCAFFOLD_TEST_SIZE, RF_N_ESTIMATORS,
    RF_MAX_DEPTH, RF_MAX_FEATURES, MAPIE_CV_FOLDS, MAPIE_CONFIDENCE, LOG_LEVEL, RUN_ID,
)
from mapie.regression import CrossConformalRegressor

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "A1": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.03,
           "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3,
           "reg_alpha": 0.1, "reg_lambda": 1.5, "gamma": 0.1},
    "A2A": {"n_estimators": 1000, "max_depth": 7, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 2,
            "reg_alpha": 0.05, "reg_lambda": 1.0, "gamma": 0.05},
    "A2B": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 2,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.05},
    "A3": {"n_estimators": 1000, "max_depth": 7, "learning_rate": 0.03,
           "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 2,
           "reg_alpha": 0.05, "reg_lambda": 1.0, "gamma": 0.05},
}

LGBM_DEFAULT_PARAMS = {
    "A1": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 40,
           "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 10,
           "reg_alpha": 0.1, "reg_lambda": 1.5},
    "A2A": {"n_estimators": 1000, "max_depth": 7, "learning_rate": 0.03, "num_leaves": 50,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 5,
            "reg_alpha": 0.05, "reg_lambda": 1.0},
    "A2B": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 40,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0},
    "A3": {"n_estimators": 1000, "max_depth": 7, "learning_rate": 0.03, "num_leaves": 50,
           "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 5,
           "reg_alpha": 0.05, "reg_lambda": 1.0},
}

def train_conformal_model(base_model, X_train, y_train, cv):
    mapie = CrossConformalRegressor(
        estimator=base_model,
        cv=cv,
        confidence_level=MAPIE_CONFIDENCE,
        method="plus",
        n_jobs=1,
    )
    mapie.fit_conformalize(X_train, y_train)
    logger.info("Conformal model trained: cv=%d, confidence=%.2f", cv, MAPIE_CONFIDENCE)
    return mapie

def _run_optuna_hpo(X_tr, y_tr, smiles_tr, subtype: str, n_trials: int = 30) -> dict:
    """
    Optuna HPO with 3-fold Scaffold-Grouped Cross-Validation (GroupKFold).
    Ensures that validation folds contain completely unseen scaffolds,
    forcing Optuna to optimize for scaffold generalizability rather than memorization.
    """
    try:
        import optuna
        from sklearn.model_selection import GroupKFold
        from src.scaffold_split import _murcko_scaffold_smiles
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Required modules for Scaffold HPO not available. Using default params.")
        return DEFAULT_PARAMS[subtype]

    # Compute scaffold for each molecule in the training set
    logger.info("  Computing Murcko scaffolds for GroupKFold HPO...")
    scaffolds = [str(_murcko_scaffold_smiles(smi)) for smi in smiles_tr]
    
    # Map scaffold strings to unique integer group IDs
    unique_scaf = list(set(scaffolds))
    scaf_to_id = {scaf: idx for idx, scaf in enumerate(unique_scaf)}
    groups = np.array([scaf_to_id[scaf] for scaf in scaffolds])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            # Restrict depth to promote generalization (scaffold-robust trees are shallower)
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
            # High L1/L2 regularization to control fingerprint overfitting
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4),
            "tree_method": "hist",
            "n_jobs": 1,  # Prevent OpenMP deadlock on Windows background runner
            "random_state": 42,
        }
        
        # 3-fold GroupKFold by scaffold
        gkf = GroupKFold(n_splits=3)
        scores = []
        for train_idx, val_idx in gkf.split(X_tr, y_tr, groups=groups):
            X_fold_tr, y_fold_tr = X_tr[train_idx], y_tr[train_idx]
            X_fold_val, y_fold_val = X_tr[val_idx], y_tr[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_fold_tr, y_fold_tr)
            preds = model.predict(X_fold_val)
            scores.append(r2_score(y_fold_val, preds))
            
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("  HPO %s (Scaffold CV): best CV R²=%.4f, params=%s", subtype, study.best_value, best)
    return best


def retrain_production_models(data_path: str = "data/raw"):
    """
    Full-dataset training with MAPIE conformal wrapping.

    CRITICAL FIX:
    - The old code saved raw XGBRegressors without conformal wrapping.
    - Now wraps every XGBoost model with CrossConformalRegressor (Jackknife+)
      to produce real uncertainty estimates.
    - The cost: slightly higher training time (5-fold CV inside MAPIE).
    - The benefit: mathematically valid 90% prediction intervals.
    """
    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING WITH MAPIE CONFORMAL WRAPPING")
    logger.info("=" * 60)



    cv_report_path = OUTPUTS_DIR / "nested_cv" / "merged_report.json"
    best_params_per_subtype = {}
    run_hpo = False

    if cv_report_path.exists():
        logger.info("Found Nested CV report at %s. Loading HPO params...", cv_report_path)
        with open(cv_report_path) as f:
            cv_data = json.load(f)
        for st in SUBTYPES:
            if st in cv_data:
                best_params_per_subtype[st] = cv_data[st]["median_params"]
                logger.info("  %s: %s", st, best_params_per_subtype[st])
    else:
        logger.info("No Nested CV HPO params found. Will run Optuna HPO per subtype.")
        run_hpo = True  # Defer to per-subtype HPO after data loading

    df, _ = load_and_clean(data_path, mode="precise", include_decoys=True)
    df = df.rename(columns={"canonical_smiles": "smiles"})

    if "barcode" in df.columns:
        logger.info("BARCODE AUDIT: %d rows, %d unique barcodes",
                     len(df), df["barcode"].nunique())

    train_smiles, test_smiles = split_smiles_globally(
        df["smiles"].unique(),
        test_size=SCAFFOLD_TEST_SIZE,
        random_state=SCAFFOLD_SPLIT_SEED,
    )

    split_path = PROCESSED_DATA_DIR / "global_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": list(train_smiles), "test": list(test_smiles)}, f)
    logger.info("Saved global scaffold split to %s", split_path)

    train_df = df[df["smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["smiles"].isin(test_smiles)].reset_index(drop=True)

    # Vectorized train_lookup building (replaces iterrows)
    train_lookup = (
        train_df.groupby("smiles")
        .apply(lambda g: dict(zip(g["target_subtype"], g["pchembl_value"].astype(float))),
               include_groups=False)
        .to_dict()
    )
    train_lookup_path = PROCESSED_DATA_DIR / "db_lookup_train.json"
    with open(train_lookup_path, "w") as f:
        json.dump(train_lookup, f, indent=2, sort_keys=True)
    logger.info("Saved train-only lookup (%d molecules) to %s", len(train_lookup), train_lookup_path)

    X_train_glob, X_test_glob, pipeline = build_feature_matrix(
        train_df, test_df, smiles_col="smiles"
    )

    models_precise_dir = MODELS_DIR / "precise"
    models_precise_dir.mkdir(parents=True, exist_ok=True)

    with open(models_precise_dir / f"scaler_precise.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Saved production scaler pipelines.")

    with open(PROCESSED_DATA_DIR / "features_train.pkl", "wb") as f:
        pickle.dump(X_train_glob, f)
    with open(PROCESSED_DATA_DIR / "features_test.pkl", "wb") as f:
        pickle.dump(X_test_glob, f)
    logger.info("Saved feature matrices.")

    training_summary = {}

    for st in SUBTYPES:
        logger.info("\nTraining %s...", st)

        train_mask = (train_df["target_subtype"] == st).values
        test_mask = (test_df["target_subtype"] == st).values

        X_tr = X_train_glob[train_mask]
        y_tr = train_df.loc[train_mask, "pchembl_value"].values

        X_te = X_test_glob[test_mask]
        y_te = test_df.loc[test_mask, "pchembl_value"].values

        if len(y_tr) < 50:
            logger.warning("SKIP %s: insufficient data (%d samples).", st, len(y_tr))
            continue

        logger.info("  Samples: train=%d, test=%d", len(y_tr), len(y_te))

        if run_hpo and st not in best_params_per_subtype:
            logger.info("  Running Optuna HPO for %s (30 trials, 3-fold Scaffold CV)...", st)
            best_params_per_subtype[st] = _run_optuna_hpo(X_tr, y_tr, train_df.loc[train_mask, "smiles"].values, st, n_trials=30)
        elif st not in best_params_per_subtype:
            best_params_per_subtype[st] = DEFAULT_PARAMS[st]

        params = best_params_per_subtype[st].copy()
        params.update({"tree_method": "hist", "n_jobs": 1, "random_state": 42})
        logger.info("  Params: %s", params)

        base_xgb = xgb.XGBRegressor(**params)

        logger.info("  Wrapping XGBoost with MAPIE CrossConformalRegressor...")
        try:
            conformal_model = train_conformal_model(
                base_model=base_xgb,
                X_train=X_tr,
                y_train=y_tr,
                cv=MAPIE_CV_FOLDS,
            )
            logger.info("  Conformal wrapping succeeded.")
        except Exception as e:
            logger.warning("  Conformal wrapping failed (%s). Saving raw model.", e)
            conformal_model = base_xgb

        rf_base = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            random_state=42,
            n_jobs=1,
        )
        logger.info("  Wrapping RandomForest with MAPIE CrossConformalRegressor...")
        try:
            rf_model = train_conformal_model(
                base_model=rf_base,
                X_train=X_tr,
                y_train=y_tr,
                cv=MAPIE_CV_FOLDS,
            )
            logger.info("  RF Conformal wrapping succeeded.")
        except Exception as e:
            logger.warning("  RF Conformal wrapping failed (%s). Saving raw model.", e)
            rf_model = rf_base
            rf_model.fit(X_tr, y_tr)

        # ── LightGBM ──
        lgb_params = LGBM_DEFAULT_PARAMS[st].copy()
        lgb_params.update({"verbosity": -1, "random_state": 42, "n_jobs": 1})
        base_lgb = lgb.LGBMRegressor(**lgb_params)
        logger.info("  Wrapping LightGBM with MAPIE CrossConformalRegressor...")
        try:
            lgb_model = train_conformal_model(
                base_model=base_lgb,
                X_train=X_tr,
                y_train=y_tr,
                cv=MAPIE_CV_FOLDS,
            )
            logger.info("  LightGBM Conformal wrapping succeeded.")
        except Exception as e:
            logger.warning("  LightGBM Conformal wrapping failed (%s). Saving raw model.", e)
            lgb_model = base_lgb
            lgb_model.fit(X_tr, y_tr)

        # ── Predictions ──
        preds_tr_xgb = conformal_model.predict(X_tr) if hasattr(conformal_model, 'predict') else base_xgb.predict(X_tr)
        preds_te_xgb = conformal_model.predict(X_te) if hasattr(conformal_model, 'predict') else base_xgb.predict(X_te)

        preds_tr_rf = rf_model.predict(X_tr)
        preds_te_rf = rf_model.predict(X_te)

        preds_tr_lgb = lgb_model.predict(X_tr)
        preds_te_lgb = lgb_model.predict(X_te)

        # ── Simple average ensemble (more robust than learned meta-model on small data) ──
        preds_tr_stack = (preds_tr_xgb + preds_tr_rf + preds_tr_lgb) / 3.0
        preds_te_stack = (preds_te_xgb + preds_te_rf + preds_te_lgb) / 3.0

        r2_tr_xgb = r2_score(y_tr, preds_tr_xgb)
        r2_te_xgb = r2_score(y_te, preds_te_xgb)
        r2_tr_rf = r2_score(y_tr, preds_tr_rf)
        r2_te_rf = r2_score(y_te, preds_te_rf)
        r2_tr_lgb = r2_score(y_tr, preds_tr_lgb)
        r2_te_lgb = r2_score(y_te, preds_te_lgb)
        r2_tr_stack = r2_score(y_tr, preds_tr_stack)
        r2_te_stack = r2_score(y_te, preds_te_stack)

        logger.info("    XGBoost  R2: train=%.3f, test=%.3f", r2_tr_xgb, r2_te_xgb)
        logger.info("    RF       R2: train=%.3f, test=%.3f", r2_tr_rf, r2_te_rf)
        logger.info("    LightGBM R2: train=%.3f, test=%.3f", r2_tr_lgb, r2_te_lgb)
        logger.info("    Stack    R2: train=%.3f, test=%.3f", r2_tr_stack, r2_te_stack)

        xgb_path = models_precise_dir / f"xgboost_{st}_production.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(conformal_model, f)

        rf_path = models_precise_dir / f"rf_{st}_production.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_model, f)

        lgb_path = models_precise_dir / f"lgb_{st}_production.pkl"
        with open(lgb_path, "wb") as f:
            pickle.dump(lgb_model, f)

        stack_model = AverageEnsemble()
        stack_path = models_precise_dir / f"stack_ridge_{st}_production.pkl"
        with open(stack_path, "wb") as f:
            pickle.dump(stack_model, f)

        training_summary[st] = {
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "xgboost_r2": r2_te_xgb,
            "rf_r2": r2_te_rf,
            "lgbm_r2": r2_te_lgb,
            "stack_r2": r2_te_stack,
            "conformal_wrapped": str(type(conformal_model).__name__),
        }
        logger.info("  Saved XGBoost model (type=%s) to %s", type(conformal_model).__name__, xgb_path)



    training_summary["run_id"] = RUN_ID
    summary_path = OUTPUTS_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    logger.info("Training summary saved to %s [run_id=%s]", summary_path, RUN_ID)

    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    retrain_production_models()
