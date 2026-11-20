"""Y-randomization permutation validation across subtypes on the global scaffold split."""
import argparse
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

from src.config import (
    LOG_LEVEL,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RUN_ID,
    SUBTYPES,
    Y_RAND_ITERATIONS,
)
from src.data_loader import load_and_clean
from src.features import build_feature_matrix

logger = logging.getLogger(__name__)


def run_y_randomization(
    subtype: str,
    data_path: str = "data/raw",
    n_iterations: int = Y_RAND_ITERATIONS,
    _cached_df=None,
    _cached_split=None,
) -> Optional[Dict[str, Any]]:
    """Run label-shuffling Y-randomization for one subtype on the held-out scaffold split."""
    logger.info("Running Y-randomization for %s (n=%d)...", subtype, n_iterations)

    if _cached_df is None:
        df, _ = load_and_clean(data_path, save_lookup_path=None, mode="precise", include_decoys=False)
    else:
        df = _cached_df

    df_st = df[df["target_subtype"] == subtype].copy().reset_index(drop=True)
    if len(df_st) < 50:
        logger.error("Insufficient data for %s (%d samples)", subtype, len(df_st))
        return None

    if _cached_split is None:
        split_path = PROCESSED_DATA_DIR / "global_split.json"
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
    else:
        split = _cached_split

    train_smiles = set(split["train"]) & set(df_st["canonical_smiles"])
    test_smiles = set(split["test"]) & set(df_st["canonical_smiles"])

    train_df = (
        df_st[df_st["canonical_smiles"].isin(train_smiles)]
        .rename(columns={"canonical_smiles": "smiles"})
        .reset_index(drop=True)
    )
    test_df = (
        df_st[df_st["canonical_smiles"].isin(test_smiles)]
        .rename(columns={"canonical_smiles": "smiles"})
        .reset_index(drop=True)
    )

    X_train, X_test, _ = build_feature_matrix(
        train_df, test_df, smiles_col="smiles", save_to_disk=False
    )
    y_train = train_df["pchembl_value"].to_numpy(dtype=float)
    y_test = test_df["pchembl_value"].to_numpy(dtype=float)

    xgb_cfg = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "n_jobs": 1,
    }

    real_model = xgb.XGBRegressor(**xgb_cfg, random_state=42)
    real_model.fit(X_train, y_train)
    real_r2 = float(r2_score(y_test, real_model.predict(X_test)))

    shuffled_r2s: list[float] = []
    for i in range(n_iterations):
        rng = np.random.RandomState(42 + i)
        y_shuffled = rng.permutation(y_train)
        shuffled_model = xgb.XGBRegressor(**xgb_cfg, random_state=42 + i)
        shuffled_model.fit(X_train, y_shuffled)
        shuf_r2 = float(r2_score(y_test, shuffled_model.predict(X_test)))
        shuffled_r2s.append(round(shuf_r2, 5))

    mean_shuf = float(np.mean(shuffled_r2s))
    std_shuf = float(np.std(shuffled_r2s))
    max_shuf = float(np.max(shuffled_r2s))
    n_ge = sum(1 for r in shuffled_r2s if r >= real_r2)
    p_value = float((n_ge + 1) / (n_iterations + 1))

    logger.info(
        "%s Y-randomization: real_r2=%.4f, shuffled=%.4f +/- %.4f (max=%.4f, p=%.4f)",
        subtype,
        real_r2,
        mean_shuf,
        std_shuf,
        max_shuf,
        p_value,
    )

    return {
        "real_r2": round(real_r2, 5),
        "shuffled_mean_r2": round(mean_shuf, 5),
        "shuffled_std_r2": round(std_shuf, 5),
        "shuffled_max_r2": round(max_shuf, 5),
        "shuffled_r2_scores": shuffled_r2s,
        "p_value": round(p_value, 5),
        "leakage_warning": bool(mean_shuf > 0.10),
    }


def run_all_subtypes(
    data_path: str = "data/raw",
    n_iterations: int = Y_RAND_ITERATIONS,
) -> Dict[str, Any]:
    """Run Y-randomization across all subtypes and write y_randomization_summary.json."""
    df, _ = load_and_clean(data_path, save_lookup_path=None, mode="precise", include_decoys=False)
    split_path = PROCESSED_DATA_DIR / "global_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    per_subtype: Dict[str, Any] = {}
    for st in SUBTYPES:
        res = run_y_randomization(
            subtype=st,
            data_path=data_path,
            n_iterations=n_iterations,
            _cached_df=df,
            _cached_split=split,
        )
        if res is not None:
            per_subtype[st] = res

    summary = {
        "run_id": RUN_ID,
        "n_iterations": int(n_iterations),
        "per_subtype": per_subtype,
    }

    out_dir = OUTPUTS_DIR / "y_randomization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "y_randomization_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved Y-randomization summary to %s", out_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Y-Randomization Permutation Validation")
    parser.add_argument("--subtype", choices=SUBTYPES, default=None, help="Single subtype to validate")
    parser.add_argument("--iterations", type=int, default=Y_RAND_ITERATIONS, help="Shuffling iterations")
    parser.add_argument("--all", action="store_true", help="Run Y-randomization across all subtypes")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()

    if args.subtype and not args.all:
        res = run_y_randomization(
            subtype=args.subtype,
            data_path=args.data_dir,
            n_iterations=args.iterations,
        )
        if res is not None:
            out_dir = OUTPUTS_DIR / "y_randomization"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{args.subtype}_y_randomization.json", "w", encoding="utf-8") as f:
                json.dump({"run_id": RUN_ID, "subtype": args.subtype, **res}, f, indent=2)
    else:
        run_all_subtypes(data_path=args.data_dir, n_iterations=args.iterations)
