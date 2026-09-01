import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.scaffold_split import scaffold_split, split_smiles_globally
from src.predictor import _load_scaler
from src.config import (
    SUBTYPES, MODELS_DIR, OUTPUTS_DIR,
    SELECTIVITY_MIN_PAIRED, SELECTIVITY_N_ESTIMATORS,
    SELECTIVITY_LR, SELECTIVITY_MAX_DEPTH,
)

logger = logging.getLogger(__name__)


def build_selectivity_models(data_path: str = "data/raw", min_paired: int = None):
    if min_paired is None:
        min_paired = SELECTIVITY_MIN_PAIRED

    logger.info("=" * 60)
    logger.info("TRAINING DIRECT SELECTIVITY MODELS (Delta-pChEMBL)")
    logger.info("=" * 60)

    # Load the production scaler pipeline fitted globally on all training data
    try:
        pipeline = _load_scaler("precise")
        logger.info("Successfully loaded the global production scaler pipeline.")
    except Exception as e:
        logger.warning("Global scaler not found: %s. Falling back to fitting scaler locally on each pair.", e)
        pipeline = None

    # 1. Load the processed DB lookup
    _, lookup = load_and_clean(data_path, mode="precise", include_decoys=True)

    # 2. Identify paired compounds for each pair
    pairs = [
        ("A2A", "A1"),   # Clinically important pairs
        ("A2A", "A3"),
        ("A1", "A3"),
        ("A1", "A2B"),
        ("A2A", "A2B"),
        ("A2B", "A3"),
    ]

    out_dir = MODELS_DIR / "selectivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    selectivity_summary = {}

    for subA, subB in pairs:
        pair_name = f"{subA}_vs_{subB}"
        logger.info("Analyzing selectivity pair: %s vs %s...", subA, subB)

        # Vectorized: collect paired data from lookup dict
        paired_data = [
            {
                "smiles": smiles,
                f"pchembl_{subA}": values[subA],
                f"pchembl_{subB}": values[subB],
                "delta_pchembl": values[subA] - values[subB],
            }
            for smiles, values in lookup.items()
            if subA in values and subB in values
        ]

        n_paired = len(paired_data)
        logger.info("  Found %d paired compounds with values for both targets.", n_paired)

        if n_paired < min_paired:
            logger.info("  SKIP: Too few paired compounds (< %d). Skipping direct model.", min_paired)
            continue

        df_pair = pd.DataFrame(paired_data)

        # 3. Scaffold split (80-20) globally at the molecule level
        train_smiles, test_smiles = split_smiles_globally(
            df_pair["smiles"].unique(), test_size=0.2, random_state=42
        )
        train_df = df_pair[df_pair["smiles"].isin(train_smiles)].reset_index(drop=True)
        test_df = df_pair[df_pair["smiles"].isin(test_smiles)].reset_index(drop=True)

        # 4. Build features using the unified global pipeline
        if pipeline is not None:
            logger.info("  Featurizing compounds using the global production pipeline...")
            X_train = np.vstack([build_features(s, pipeline) for s in train_df["smiles"]])
            X_test = np.vstack([build_features(s, pipeline) for s in test_df["smiles"]])
            pair_pipeline = pipeline
        else:
            logger.warning("  Falling back to fitting pair-specific scaler pipeline...")
            X_train, X_test, pair_pipeline = build_feature_matrix(
                train_df, test_df, smiles_col="smiles", save_to_disk=False
            )

        y_train = train_df["delta_pchembl"].values
        y_test = test_df["delta_pchembl"].values

        # 5. Train XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=SELECTIVITY_N_ESTIMATORS,
            learning_rate=SELECTIVITY_LR,
            max_depth=SELECTIVITY_MAX_DEPTH,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 6. Evaluate
        preds = model.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

        logger.info("  Evaluation Results for %s:", pair_name)
        logger.info("    R² Score: %.4f", r2)
        logger.info("    MAE:      %.4f pChEMBL units", mae)
        logger.info("    RMSE:     %.4f pChEMBL units", rmse)

        # Save model and pipeline
        model_path = out_dir / f"xgb_selectivity_{pair_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        pipeline_path = out_dir / f"xgb_selectivity_{pair_name}_pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pair_pipeline, f)

        selectivity_summary[pair_name] = {
            "n_paired": n_paired,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        }

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(selectivity_summary, f, indent=2)
    logger.info("Wrote direct selectivity summary to %s", summary_file)


def predict_direct_selectivity(smiles: str, subA: str, subB: str) -> float | None:
    """
    Predict direct selectivity difference (pChEMBL_subA - pChEMBL_subB) for a given compound SMILES.
    """
    pair_name = f"{subA}_vs_{subB}"
    model_path = MODELS_DIR / "selectivity" / f"xgb_selectivity_{pair_name}_model.pkl"
    pipeline_path = MODELS_DIR / "selectivity" / f"xgb_selectivity_{pair_name}_pipeline.pkl"

    if not model_path.exists() or not pipeline_path.exists():
        # Fallback to reverse pair name if exists
        reverse_pair_name = f"{subB}_vs_{subA}"
        rev_model_path = MODELS_DIR / "selectivity" / f"xgb_selectivity_{reverse_pair_name}_model.pkl"
        rev_pipeline_path = MODELS_DIR / "selectivity" / f"xgb_selectivity_{reverse_pair_name}_pipeline.pkl"

        if rev_model_path.exists() and rev_pipeline_path.exists():
            with open(rev_model_path, "rb") as f:
                model = pickle.load(f)
            with open(rev_pipeline_path, "rb") as f:
                pipeline = pickle.load(f)
            x = build_features(smiles, pipeline)
            pred = model.predict(x.reshape(1, -1))[0]
            return -float(pred)  # Negate since order is reversed
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    x = build_features(smiles, pipeline)
    pred = model.predict(x.reshape(1, -1))[0]
    return float(pred)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    build_selectivity_models()
