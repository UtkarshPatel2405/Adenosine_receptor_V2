import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from src.data_loader import load_and_clean
from src.features import build_feature_matrix, build_features
from src.config import SUBTYPES

# Suppress Optuna logging to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _get_scaffold_folds(df: pd.DataFrame, smiles_col: str = "canonical_smiles",
                        n_splits: int = 5, random_state: int = 42):
    """Group indices by Murcko scaffold and distribute round-robin into folds."""
    from src.scaffold_split import _murcko_scaffold_smiles
    scaffolds = df[smiles_col].apply(_murcko_scaffold_smiles).values
    
    scaffold_to_indices = {}
    for idx, scaf in enumerate(scaffolds):
        scaffold_to_indices.setdefault(scaf, []).append(idx)
        
    import random
    rng = random.Random(random_state)
    scaf_keys = sorted(list(scaffold_to_indices.keys()))
    rng.shuffle(scaf_keys)
    
    folds_indices = [[] for _ in range(n_splits)]
    for idx_k, scaf in enumerate(scaf_keys):
        folds_indices[idx_k % n_splits].extend(scaffold_to_indices[scaf])
        
    return folds_indices

def run_fold(subtype: str, fold: int, trials: int = 20, data_path: str = "data/raw/AR_all_unique_parents_with_smiles.csv"):
    print(f"\n[INFO] Starting Nested CV: Subtype={subtype}, Fold={fold}/5 (trials={trials})")
    
    # 1. Load precise cleaned data
    df, _ = load_and_clean(data_path, mode="precise")
    
    # Filter for this subtype
    df_st = df[df["target_subtype"] == subtype].copy().reset_index(drop=True)
    if len(df_st) < 50:
        print(f"[ERROR] Insufficient data for {subtype} ({len(df_st)} samples)")
        return
        
    # Standardize column name
    df_st = df_st.rename(columns={"canonical_smiles": "smiles"})
    
    # 2. Get deterministic outer folds
    outer_folds = _get_scaffold_folds(df_st, smiles_col="smiles", n_splits=5, random_state=42)
    test_idx = outer_folds[fold - 1]
    train_idx = []
    for f in range(5):
        if f != fold - 1:
            train_idx.extend(outer_folds[f])
            
    train_df = df_st.iloc[train_idx].reset_index(drop=True)
    test_df = df_st.iloc[test_idx].reset_index(drop=True)
    
    print(f"[INFO] Dataset split: Train={len(train_df)} compounds, Test={len(test_df)} compounds")
    
    # 3. Build features globally on this outer train/test split
    X_train, X_test, pipeline = build_feature_matrix(train_df, test_df, smiles_col="smiles")
    y_train = train_df["pchembl_value"].values
    y_test = test_df["pchembl_value"].values
    
    # 4. Optuna HPO Inner Loop (3-fold scaffold split CV)
    inner_folds = _get_scaffold_folds(train_df, smiles_col="smiles", n_splits=3, random_state=100 + fold)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42
        }
        
        inner_maes = []
        for inner_f in range(3):
            val_idx_inner = inner_folds[inner_f]
            train_idx_inner = []
            for inf in range(3):
                if inf != inner_f:
                    train_idx_inner.extend(inner_folds[inf])
                    
            X_tr_inner, y_tr_inner = X_train[train_idx_inner], y_train[train_idx_inner]
            X_val_inner, y_val_inner = X_train[val_idx_inner], y_train[val_idx_inner]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr_inner, y_tr_inner)
            preds = model.predict(X_val_inner)
            mae = np.mean(np.abs(y_val_inner - preds))
            inner_maes.append(mae)
            
        return np.mean(inner_maes)
        
    print(f"[INFO] Launching Optuna with {trials} trials on inner loop...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    
    best_params = study.best_params
    print(f"[SUCCESS] Fold {fold} HPO complete. Best inner MAE: {study.best_value:.4f}")
    print(f"[INFO] Best parameters: {best_params}")
    
    # 5. Train final model on full outer train set with best parameters
    best_params.update({"tree_method": "hist", "n_jobs": -1, "random_state": 42})
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on outer test set
    preds = final_model.predict(X_test)
    
    # Calculate metrics
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    from sklearn.metrics import r2_score
    r2 = float(r2_score(y_test, preds))
    
    print(f"[RESULTS] Outer Fold {fold} Evaluation:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  RMSE:      {rmse:.4f}")
    
    # Save fold output to disk
    out_dir = Path("outputs/nested_cv")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fold_data = {
        "subtype": subtype,
        "fold": fold,
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "best_params": study.best_params,
        "predictions": [float(p) for p in preds],
        "actuals": [float(a) for a in y_test],
        "smiles": test_df["smiles"].tolist()
    }
    
    fold_file = out_dir / f"{subtype}_fold{fold}.json"
    with open(fold_file, "w") as f:
        json.dump(fold_data, f, indent=2)
    print(f"[SUCCESS] Saved fold results to {fold_file}")


def merge_results():
    print("\n" + "="*50)
    print("MERGING NESTED CV RESULTS")
    print("="*50)
    
    out_dir = Path("outputs/nested_cv")
    if not out_dir.exists():
        print("[ERROR] outputs/nested_cv directory does not exist! No results to merge.")
        return
        
    files = list(out_dir.glob("*_fold*.json"))
    if not files:
        print("[ERROR] No fold results found in outputs/nested_cv/")
        return
        
    results_by_subtype = {}
    for f in files:
        try:
            with open(f, "r") as json_f:
                data = json.load(json_f)
            subtype = data["subtype"]
            results_by_subtype.setdefault(subtype, []).append(data)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            
    print(f"[INFO] Found results for subtypes: {list(results_by_subtype.keys())}")
    
    merged_summary = {}
    
    for subtype, folds_data in results_by_subtype.items():
        n_folds = len(folds_data)
        print(f"\nProcessing {subtype} ({n_folds}/5 folds completed)...")
        
        # Calculate stats across folds
        r2s = [f["metrics"]["r2"] for f in folds_data]
        maes = [f["metrics"]["mae"] for f in folds_data]
        rmses = [f["metrics"]["rmse"] for f in folds_data]
        
        mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
        mean_mae, std_mae = np.mean(maes), np.std(maes)
        mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
        
        print(f"  R²  = {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  MAE  = {mean_mae:.4f} ± {std_mae:.4f}")
        print(f"  RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        # Determine median hyperparameters across folds to use for production
        all_params = [f["best_params"] for f in folds_data]
        median_params = {}
        for param_key in all_params[0].keys():
            param_vals = [p[param_key] for p in all_params]
            if isinstance(param_vals[0], (int, float)):
                median_params[param_key] = float(np.median(param_vals)) if isinstance(param_vals[0], float) else int(np.round(np.median(param_vals)))
            else:
                median_params[param_key] = param_vals[0]
                
        print(f"  Production Hyperparameters (median): {median_params}")
        
        merged_summary[subtype] = {
            "n_folds_completed": n_folds,
            "metrics": {
                "r2_mean": float(mean_r2), "r2_std": float(std_r2),
                "mae_mean": float(mean_mae), "mae_std": float(std_mae),
                "rmse_mean": float(mean_rmse), "rmse_std": float(std_rmse)
            },
            "median_params": median_params
        }
        
    # Save the merged report to disk
    merged_file = out_dir / "merged_report.json"
    with open(merged_file, "w") as f:
        json.dump(merged_summary, f, indent=2)
    print(f"\n[SUCCESS] Wrote merged json report to {merged_file}")
    
    # Write a beautiful markdown file
    md_file = out_dir / "merged_report.md"
    with open(md_file, "w") as f:
        f.write("# Nested Cross-Validation Performance Summary\n\n")
        f.write("This report aggregates results from the chunk-and-merge Nested Cross-Validation runs using a scaffold split.\n\n")
        f.write("| Subtype | Completed Folds | R² Score | MAE | RMSE |\n")
        f.write("|---|---|---|---|---|\n")
        for st in SUBTYPES:
            if st in merged_summary:
                m = merged_summary[st]["metrics"]
                nf = merged_summary[st]["n_folds_completed"]
                f.write(f"| **{st}** | {nf}/5 | {m['r2_mean']:.3f} ± {m['r2_std']:.3f} | {m['mae_mean']:.3f} ± {m['mae_std']:.3f} | {m['rmse_mean']:.3f} ± {m['rmse_std']:.3f} |\n")
            else:
                f.write(f"| **{st}** | 0/5 | Not Started | N/A | N/A |\n")
                
        f.write("\n\n## Best Median Hyperparameters for Production Training\n\n")
        for st in SUBTYPES:
            if st in merged_summary:
                f.write(f"### Subtype {st}\n")
                f.write("```json\n")
                f.write(json.dumps(merged_summary[st]["median_params"], indent=2))
                f.write("\n```\n\n")
                
    print(f"[SUCCESS] Wrote merged markdown report to {md_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk-and-Merge Nested Cross-Validation with Optuna HPO")
    parser.add_argument("--subtype", choices=SUBTYPES, help="Target subtype to run")
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], help="Outer fold index (1 to 5)")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna HPO trials in the inner loop")
    parser.add_argument("--merge", action="store_true", help="Merge all completed fold results and write report")
    
    args = parser.parse_args()
    
    if args.merge:
        merge_results()
    elif args.subtype and args.fold:
        run_fold(subtype=args.subtype, fold=args.fold, trials=args.trials)
    else:
        parser.print_help()
