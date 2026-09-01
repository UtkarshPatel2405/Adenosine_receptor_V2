"""
SHAP Tree Explainability — Global and local feature attribution analysis.

Now supports ALL 4 subtypes via --all flag.
"""

import os
import pickle
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.predictor import SUBTYPES

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def run_shap_analysis(subtype: str = "A2A", mode: str = "precise"):
    logger.info("=" * 60)
    logger.info("SHAP TREE ANALYSIS FOR %s (%s mode)", subtype, mode)
    logger.info("=" * 60)
    
    # 1. Load trained model
    model_path = Path(f"models/{mode}/xgboost_{subtype}_production.pkl")
    if not model_path.exists():
        model_path = Path(f"models/{mode}/xgboost_{mode}_{subtype.lower()}_model.pkl")
    if not model_path.exists():
        model_path = Path(f"models/xgboost_{subtype.lower()}_model.pkl")
    if not model_path.exists():
        model_path = Path(f"models/xgboost_precise_{subtype.lower()}_model.pkl")
    if not model_path.exists():
        logger.error("No model found for %s. Complete production retraining first.", subtype)
        return None

    logger.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Extract fitted base estimator for SHAP TreeExplainer
    # MAPIE 1.4.1 CrossConformalRegressor stores CV estimators in _mapie_regressor
    model_type = type(model).__name__
    if model_type in ("CrossConformalRegressor", "MapieRegressor"):
        reg = model._mapie_regressor.estimator_ if model_type == "CrossConformalRegressor" else model.estimator_
        if hasattr(reg, "single_estimator_"):
            estimator = reg.single_estimator_
        elif hasattr(reg, "estimators_") and len(reg.estimators_) > 0:
            estimator = reg.estimators_[0]
        else:
            estimator = reg
    elif isinstance(model, list) and len(model) > 0:
        estimator = model[0]
    else:
        estimator = model
        
    # 2. Load test features and feature names using the saved split
    split_path = Path("data/processed/global_split.json")
    if not split_path.exists():
        logger.error("Global split missing. Run retrain_production.py first.")
        return None
        
    with open(split_path, "r") as f:
        splits = json.load(f)
        
    from src.data_loader import load_and_clean
    # Load test dataframe
    df, _ = load_and_clean("data/raw", mode="precise", include_decoys=True)
    test_df_full = df[df["canonical_smiles"].isin(splits["test"])].reset_index(drop=True)
    
    # Load global X_test
    with open("data/processed/features_test.pkl", "rb") as f:
        X_test_glob = pickle.load(f)
        
    subtype_mask = (test_df_full["target_subtype"] == subtype).values
    X_test = X_test_glob[subtype_mask]
    test_df = test_df_full[subtype_mask].reset_index(drop=True)
    
    scaler_path = Path("models/scaler.pkl")
    if not scaler_path.exists():
        logger.error("Scaler missing.")
        return None
        
    with open(scaler_path, "rb") as f:
        pipeline = pickle.load(f)
        
    # Build complete feature names list
    feature_names = [f"Morgan_FP_{i}" for i in range(2048)] + [f"MACCS_{i}" for i in range(167)]
    selected_desc_names = pipeline.feature_filter.feature_names
    feature_names.extend(selected_desc_names)
    
    # Trim feature names to match actual X_test shape
    if len(feature_names) > X_test.shape[1]:
        feature_names = feature_names[:X_test.shape[1]]
    elif len(feature_names) < X_test.shape[1]:
        for i in range(len(feature_names), X_test.shape[1]):
            feature_names.append(f"Feature_{i}")
    
    logger.info("Loaded test set with shape: %s", X_test.shape)
    logger.info("  Total features: %d", len(feature_names))
    
    # 3. Create TreeExplainer
    logger.info("Initializing SHAP TreeExplainer and calculating SHAP values...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_test)
    
    out_dir = Path("outputs/shap")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Global Feature Importance (Beeswarm): {subtype}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    beeswarm_file = out_dir / f"{subtype}_beeswarm.png"
    plt.savefig(beeswarm_file, dpi=300)
    plt.close()
    logger.info("Saved summary beeswarm plot to %s", beeswarm_file)
    
    # Generate Bar plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Global Feature Importance (Bar): {subtype}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    bar_file = out_dir / f"{subtype}_bar.png"
    plt.savefig(bar_file, dpi=300)
    plt.close()
    logger.info("Saved summary bar plot to %s", bar_file)
    
    # Analyze the most important features to perform the Chemical Sanity Check
    mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shaps)[::-1]
    
    top_features = []
    logger.info("Chemical Sanity Check - Top 10 Most Important Features for %s:", subtype)
    for rank, idx in enumerate(sorted_indices[:10], 1):
        name = feature_names[idx]
        val = mean_abs_shaps[idx]
        top_features.append({"rank": rank, "feature": name, "mean_abs_shap": float(val)})
        logger.info("  %d. %s: %.4f", rank, name, val)
        
    # Standard chemical analysis for Adenosine Receptor selectivities
    expected_top = ["AromRings", "HBD", "HBA", "TPSA", "LogP", "MolWt", "NumAromaticRings", "NumHDonors", "NumHAcceptors", "MolLogP"]
    matching_expected = [f for f in expected_top if any(f in item["feature"] for item in top_features)]
    
    sanity_status = "PASS"
    sanity_message = "Top features are dominated by structurally relevant physicochemical properties."
    if len(matching_expected) == 0:
        sanity_status = "WARNING"
        sanity_message = "Top features are dominated by isolated fingerprint bits rather than global physicochemical descriptors. Verify model is not learning noise."
        
    logger.info("Chemical Sanity for %s: %s - %s", subtype, sanity_status, sanity_message)
    
    report_data = {
        "subtype": subtype,
        "mode": mode,
        "top_features": top_features,
        "sanity_check": {
            "status": sanity_status,
            "message": sanity_message,
            "expected_features_found": matching_expected
        }
    }
    
    with open(out_dir / f"{subtype}_shap_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved SHAP report to %s/%s_shap_report.json", out_dir, subtype)
    
    return report_data


def run_all_subtypes(mode: str = "precise"):
    """Run SHAP analysis for ALL 4 receptor subtypes."""
    logger.info("=" * 60)
    logger.info("SHAP TREE ANALYSIS FOR ALL SUBTYPES")
    logger.info("=" * 60)
    
    all_results = {}
    for st in SUBTYPES:
        result = run_shap_analysis(subtype=st, mode=mode)
        if result is not None:
            all_results[st] = result
    
    # Save combined summary
    out_dir = Path("outputs/shap")
    summary = {
        "n_subtypes_analyzed": len(all_results),
        "subtypes": list(all_results.keys()),
        "sanity_summary": {
            st: r["sanity_check"]["status"] for st, r in all_results.items()
        }
    }
    with open(out_dir / "all_subtypes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("All-subtype SHAP summary saved to %s", out_dir / 'all_subtypes_summary.json')
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHAP Tree Explainer validation")
    parser.add_argument("--subtype", default=None, help="Subtype to explain")
    parser.add_argument("--mode", default="precise", help="Model mode")
    parser.add_argument("--all", action="store_true", help="Run SHAP for ALL 4 subtypes")
    args = parser.parse_args()
    
    if args.all:
        run_all_subtypes(mode=args.mode)
    elif args.subtype:
        run_shap_analysis(subtype=args.subtype, mode=args.mode)
    else:
        # Default: run all subtypes
        run_all_subtypes(mode=args.mode)
