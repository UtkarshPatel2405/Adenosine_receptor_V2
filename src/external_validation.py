"""
External Validation — Blind test using GPCRdb data not seen during training.

Loads GPCRdb Excel files, filters out training compounds via barcode registry,
and runs predictions on truly external molecules.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


from src.predictor import predict, SUBTYPES
from src.chem_utils import canonicalize as _canonicalize




def prepare_external_test_set(
    gpcrdb_files: dict = None,
    train_lookup_path: str = "data/processed/db_lookup.json",
) -> pd.DataFrame:
    """
    Prepare an external blind validation set from GPCRdb Excel files.
    Filters out any compound seen during training.
    """
    if gpcrdb_files is None:
        gpcrdb_files = {
            "A1": "data/raw/GPCRdb_A1.xlsx",
            "A2A": "data/raw/GPCRdb_A2A.xlsx",
            "A2B": "data/raw/GPCRdb_A2B.xlsx",
            "A3": "data/raw/GPCRdb_A3.xlsx",
        }
    
    # Load training compound lookup
    train_lookup = {}
    if Path(train_lookup_path).exists():
        with open(train_lookup_path, "r") as f:
            train_lookup = json.load(f)
    
    seen_smiles = set(train_lookup.keys())
    print(f"[INFO] Training set contains {len(seen_smiles)} unique canonical SMILES.")
    
    # Collect novel molecules from GPCRdb
    novel_molecules = {}
    
    for subtype, filepath in gpcrdb_files.items():
        if not Path(filepath).exists():
            print(f"[WARNING] GPCRdb file not found: {filepath}. Skipping {subtype}.")
            continue
        
        df = pd.read_excel(filepath)
        
        # Try to find SMILES and pChEMBL columns
        smiles_col = None
        pval_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if "smiles" in col_lower:
                smiles_col = col
            if "p-value" in col_lower or "pchembl" in col_lower or "p_value" in col_lower:
                pval_col = col
        
        if smiles_col is None or pval_col is None:
            print(f"[WARNING] Could not find SMILES/pValue columns in {filepath}. Columns: {list(df.columns)}")
            continue
        
        n_total = 0
        n_novel = 0
        n_seen = 0
        
        for _, row in df.iterrows():
            smi = row.get(smiles_col)
            p_val = row.get(pval_col)
            
            if pd.isna(smi) or pd.isna(p_val):
                continue
            
            canon = _canonicalize(str(smi))
            if canon is None:
                continue
            
            n_total += 1
            
            if canon in seen_smiles:
                n_seen += 1
                continue  # Data leakage prevention
            
            n_novel += 1
            
            if canon not in novel_molecules:
                novel_molecules[canon] = {"canonical_smiles": canon}
            
            # Keep the max pChEMBL if multiple entries exist
            current_val = novel_molecules[canon].get(subtype, 0)
            try:
                novel_molecules[canon][subtype] = max(current_val, float(p_val))
            except (ValueError, TypeError):
                pass
        
        print(f"  {subtype}: {n_total} total, {n_novel} novel, {n_seen} seen (filtered out)")
    
    # Convert to DataFrame
    rows = []
    for canon, data in novel_molecules.items():
        row = {"canonical_smiles": canon}
        for st in SUBTYPES:
            if st in data:
                row[st] = data[st]
        rows.append(row)
    
    novel_df = pd.DataFrame(rows)
    print(f"\n[INFO] External validation set: {len(novel_df)} novel molecules")
    
    # Save
    out_path = Path("data/processed/novel_test_set.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    novel_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved external test set to {out_path}")
    
    return novel_df


def run_external_validation(novel_df: pd.DataFrame = None, mode: str = "precise") -> dict:
    """Run predictions on external test set and compute performance metrics."""
    
    if novel_df is None:
        csv_path = Path("data/processed/novel_test_set.csv")
        if csv_path.exists():
            novel_df = pd.read_csv(csv_path)
        else:
            novel_df = prepare_external_test_set()
    
    if len(novel_df) == 0:
        print("[ERROR] No novel molecules for external validation.")
        return {}
    
    print(f"\n{'='*60}")
    print(f"EXTERNAL VALIDATION ({len(novel_df)} novel molecules)")
    print(f"{'='*60}")
    
    results = []
    errors = 0
    
    for idx, row in novel_df.iterrows():
        smiles = row["canonical_smiles"]
        try:
            pred = predict(smiles)
            result_row = {"canonical_smiles": smiles}
            for st in SUBTYPES:
                try:
                    result_row[f"{st}_pred"] = pred["predictions"]["XGBoost"][st]
                except KeyError:
                    result_row[f"{st}_pred"] = np.nan
                result_row[f"{st}_true"] = row.get(st, np.nan)
            results.append(result_row)
        except Exception as e:
            print(f"Error predicting {smiles}: {e}")
            errors += 1
            
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(novel_df)}] Predictions completed...")
    
    if not results:
        print("[ERROR] No successful predictions.")
        return {}
    
    result_df = pd.DataFrame(results)
    
    # Save predictions
    out_path = Path("data/processed/novel_test_result.csv")
    result_df.to_csv(out_path, index=False)
    
    # Per-subtype metrics
    metrics = {}
    for st in SUBTYPES:
        true_col = f"{st}_true"
        pred_col = f"{st}_pred"
        
        if true_col not in result_df.columns:
            continue
        
        mask = result_df[true_col].notna() & result_df[pred_col].notna()
        n = mask.sum()
        
        if n < 5:
            metrics[st] = {"n": int(n), "insufficient_data": True}
            continue
        
        y_true = result_df.loc[mask, true_col].values
        y_pred = result_df.loc[mask, pred_col].values
        
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        
        metrics[st] = {
            "n": int(n),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        }
        print(f"  {st}: n={n}, R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Selectivity accuracy (Recall@1)
    subtypes_with_truth = [st for st in SUBTYPES if f"{st}_true" in result_df.columns]
    
    if len(subtypes_with_truth) >= 2:
        multi_target = result_df.dropna(subset=[f"{st}_true" for st in subtypes_with_truth], thresh=2)
        correct = 0
        total = 0
        
        for _, row in multi_target.iterrows():
            true_vals = {st: row[f"{st}_true"] for st in subtypes_with_truth if pd.notna(row.get(f"{st}_true"))}
            pred_vals = {st: row[f"{st}_pred"] for st in true_vals.keys() if pd.notna(row.get(f"{st}_pred"))}
            
            if len(true_vals) >= 2 and len(pred_vals) >= 2:
                true_best = max(true_vals, key=true_vals.get)
                pred_best = max(pred_vals, key=pred_vals.get)
                total += 1
                if true_best == pred_best:
                    correct += 1
        
        selectivity_accuracy = correct / total if total > 0 else 0.0
        print(f"\n  Selectivity Recall@1: {correct}/{total} = {selectivity_accuracy:.3f}")
        metrics["selectivity_recall_at_1"] = {
            "correct": correct,
            "total": total,
            "accuracy": selectivity_accuracy,
        }
    
    # Save report with run_id
    from src.config import RUN_ID, OUTPUTS_DIR
    from src.run_id import save_with_run_id

    metrics["run_id"] = RUN_ID
    out_dir = OUTPUTS_DIR / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_with_run_id(metrics, out_dir, "external_validation_report", RUN_ID)
    print(f"\n[SUCCESS] External validation report saved to {out_dir} with run ID {RUN_ID}")
    
    return metrics


if __name__ == "__main__":
    novel_df = prepare_external_test_set()
    if len(novel_df) > 0:
        run_external_validation(novel_df)
