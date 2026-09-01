"""
Y-Randomization Validation — Label shuffling test to prove genuine chemical SAR.

Now supports ALL 4 subtypes (A1, A2A, A2B, A3) via --all flag.
"""

import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score

from src.data_loader import load_and_clean
from src.features import build_feature_matrix
from src.config import SUBTYPES, Y_RAND_ITERATIONS

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def run_y_randomization(subtype: str, data_path: str = "data/raw", n_iterations: int = 20):
    logger.info("=" * 60)
    logger.info("Y-RANDOMIZATION TEST FOR %s (n=%d)", subtype, n_iterations)
    logger.info("=" * 60)

    df, _ = load_and_clean(data_path, mode="precise",
                            save_lookup_path="data/processed/db_lookup_actives_only.json",
                            include_decoys=False)
    df_st = df[df["target_subtype"] == subtype].copy().reset_index(drop=True)

    if len(df_st) < 50:
        logger.error("Insufficient data for %s (%d samples)", subtype, len(df_st))
        return None

    split_path = Path("data/processed/global_split.json")
    with open(split_path) as f:
        split = json.load(f)
    train_smiles = set(split["train"])
    test_smiles = set(split["test"])

    available_smiles = set(df_st["canonical_smiles"])
    train_smiles = train_smiles & available_smiles
    test_smiles = test_smiles & available_smiles

    df_st_split = df_st[df_st["canonical_smiles"].isin(train_smiles | test_smiles)]
    train_df = df_st_split[df_st_split["canonical_smiles"].isin(train_smiles)].copy()
    test_df = df_st_split[df_st_split["canonical_smiles"].isin(test_smiles)].copy()

    train_df = train_df.rename(columns={"canonical_smiles": "smiles"})
    test_df = test_df.rename(columns={"canonical_smiles": "smiles"})

    X_train, X_test, _ = build_feature_matrix(train_df, test_df, smiles_col="smiles", save_to_disk=False)
    y_train = train_df["pchembl_value"].values
    y_test = test_df["pchembl_value"].values

    base_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=42
    )
    base_model.fit(X_train, y_train)
    real_preds = base_model.predict(X_test)
    real_r2 = float(r2_score(y_test, real_preds))
    logger.info("Real Model R² Score: %.4f", real_r2)

    shuffled_r2s = []

    for i in range(n_iterations):
        y_train_shuffled = np.random.RandomState(42 + i).permutation(y_train)

        shuffled_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=42 + i
        )
        shuffled_model.fit(X_train, y_train_shuffled)

        shuffled_preds = shuffled_model.predict(X_test)
        shuffled_r2 = float(r2_score(y_test, shuffled_preds))
        shuffled_r2s.append(shuffled_r2)

        if (i + 1) % 5 == 0:
            logger.info("  Iteration %d/%d | R² = %.4f", i + 1, n_iterations, shuffled_r2)

    mean_shuffled_r2 = float(np.mean(shuffled_r2s))
    std_shuffled_r2 = float(np.std(shuffled_r2s))

    logger.info("Y-Randomization Summary for %s:", subtype)
    logger.info("  Real R²:         %.4f", real_r2)
    logger.info("  Shuffled R²:     %.4f ± %.4f", mean_shuffled_r2, std_shuffled_r2)

    leakage_flag = False
    if mean_shuffled_r2 > 0.10:
        logger.warning("Shuffled R² > 0.10! Possible target leakage!")
        leakage_flag = True
    else:
        logger.info("Shuffled R² near-zero. Model relies on real chemical structures.")

    out_dir = Path("outputs/y_randomization")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "subtype": subtype, "n_iterations": n_iterations,
        "real_r2": real_r2, "shuffled_r2_mean": mean_shuffled_r2,
        "shuffled_r2_std": std_shuffled_r2, "shuffled_r2_values": shuffled_r2s,
        "leakage_warning": leakage_flag,
    }

    with open(out_dir / f"{subtype}_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(shuffled_r2s, fill=True, label="Shuffled Labels R²", color="skyblue", lw=2)
    plt.axvline(real_r2, color="crimson", linestyle="--", lw=2.5, label=f"Real Model R² ({real_r2:.3f})")
    plt.axvline(0.0, color="gray", linestyle="-", alpha=0.5)
    plt.title(f"Y-Randomization Test: {subtype} subtype", fontsize=13, fontweight="bold")
    plt.xlabel("R² Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plot_file = out_dir / f"{subtype}_distribution.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logger.info("Saved Y-Randomization distribution plot to %s", plot_file)

    return report_data


def run_all_subtypes(n_iterations: int = 20):
    """Run Y-randomization for ALL 4 receptor subtypes."""
    logger.info("=" * 60)
    logger.info("Y-RANDOMIZATION VALIDATION FOR ALL SUBTYPES")
    logger.info("=" * 60)

    all_results = {}
    for st in SUBTYPES:
        result = run_y_randomization(subtype=st, n_iterations=n_iterations)
        if result is not None:
            all_results[st] = result

    out_dir = Path("outputs/y_randomization")
    summary = {
        "n_subtypes_validated": len(all_results),
        "subtypes": list(all_results.keys()),
        "summary": {
            st: {"real_r2": r["real_r2"], "shuffled_r2_mean": r["shuffled_r2_mean"],
                 "shuffled_r2_std": r["shuffled_r2_std"], "leakage_warning": r["leakage_warning"]}
            for st, r in all_results.items()
        },
    }
    with open(out_dir / "all_subtypes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("All-subtype Y-randomization summary saved.")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Y-Randomization validation test")
    parser.add_argument("--subtype", default=None, help="Subtype to validate (or use --all)")
    parser.add_argument("--iterations", type=int, default=20, help="Number of shuffling runs")
    parser.add_argument("--all", action="store_true", help="Run Y-randomization for ALL 4 subtypes")
    args = parser.parse_args()
    
    if args.all:
        run_all_subtypes(n_iterations=args.iterations)
    elif args.subtype:
        run_y_randomization(subtype=args.subtype, n_iterations=args.iterations)
    else:
        # Default: run all subtypes
        run_all_subtypes(n_iterations=args.iterations)
