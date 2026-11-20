"""Holdout scaffold evaluation, calibration diagnostics, and dataset statistics."""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    LOG_LEVEL,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RUN_ID,
    SCAFFOLD_SPLIT_SEED,
    SCAFFOLD_TEST_SIZE,
    SUBTYPES,
)
from src.data_loader import load_all_raw_data, load_and_clean
from src.features import _all_descriptors, _maccs_bits, _morgan_bits
from src.models.ensemble_engine import _ensemble_predict
from src.models.model_loader import (
    _load_consensus_weights,
    _load_lgb_models,
    _load_rf_models,
    _load_scaler,
    _load_xgb_models,
)
from src.scaffold_split import _murcko_scaffold_smiles

logger = logging.getLogger(__name__)


def _safe_spearman(y_true: np.ndarray, y_pred: Optional[np.ndarray]) -> Optional[float]:
    if y_pred is None or len(y_true) < 2 or len(y_pred) != len(y_true):
        return None
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return None
    rho, _ = spearmanr(y_true, y_pred)
    return None if np.isnan(rho) else float(rho)


def save_with_run_id(
    data: dict,
    out_dir_or_path: str | Path,
    filename_or_runid: str,
    run_id: Optional[str] = None,
) -> None:
    """Save data to both a run-id-prefixed JSON path and a canonical JSON path."""
    if run_id is None:
        run_id = filename_or_runid
        target = Path(out_dir_or_path)
    else:
        target = Path(out_dir_or_path) / filename_or_runid

    target.parent.mkdir(parents=True, exist_ok=True)
    target_file = target if str(target).endswith(".json") else Path(f"{target}.json")
    versioned = target_file.parent / f"{run_id}_{target_file.name}"

    save_data = dict(data)
    save_data["run_id"] = run_id

    with open(versioned, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)


def _calibration_quartiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    lowers: Optional[np.ndarray] = None,
    uppers: Optional[np.ndarray] = None,
) -> List[dict]:
    """Group test predictions into uncertainty quartiles and compute MAE and empirical coverage."""
    n = len(y_true)
    if n < 8:
        return []

    order = np.argsort(y_std)
    bins = np.array_split(order, 4)

    out = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        item = {
            "bin": i,
            "n": int(len(idx)),
            "std_mean": float(np.mean(y_std[idx])),
            "mae_mean": float(np.mean(np.abs(y_true[idx] - y_pred[idx]))),
        }
        if lowers is not None and uppers is not None:
            item["coverage_90"] = float(np.mean((y_true[idx] >= lowers[idx]) & (y_true[idx] <= uppers[idx])))
        out.append(item)
    return out


def _transform_df(df: pd.DataFrame, pipeline) -> np.ndarray:
    from joblib import Parallel, delayed

    smiles = df["canonical_smiles"].tolist()
    unique_smiles = list(dict.fromkeys(smiles))
    u_fp = np.vstack(Parallel(n_jobs=-1)(delayed(_morgan_bits)(s) for s in unique_smiles))
    u_maccs = np.vstack(Parallel(n_jobs=-1)(delayed(_maccs_bits)(s) for s in unique_smiles))
    u_desc = np.vstack(Parallel(n_jobs=-1)(delayed(_all_descriptors)(s) for s in unique_smiles))

    idx_map = {s: i for i, s in enumerate(unique_smiles)}
    rows_idx = [idx_map[s] for s in smiles]
    X_desc_s = pipeline.transform(u_desc[rows_idx])
    return np.hstack([u_fp[rows_idx], u_maccs[rows_idx], X_desc_s]).astype(np.float32)


def _compute_dataset_diagnostics(df: pd.DataFrame, data_path: str, run_id: str) -> dict:
    raw_sources = df.attrs.get("raw_sources_breakdown")
    n_raw_records = df.attrs.get("n_raw_records")
    if raw_sources is None or n_raw_records is None:
        raw_df = load_all_raw_data(data_path)
        n_raw_records = int(len(raw_df))
        raw_sources = (
            {str(k): int(v) for k, v in raw_df["source_db"].value_counts().items()}
            if "source_db" in raw_df.columns
            else {}
        )

    n_replicates_dropped_conflict = int(df.attrs.get("n_replicates_dropped_conflict", 0))
    unique_smiles = df["canonical_smiles"].unique().tolist()
    subtype_counts_per_mol = df.groupby("canonical_smiles")["target_subtype"].nunique()
    n_multi_subtype = int((subtype_counts_per_mol >= 2).sum())

    scaffolds = {_murcko_scaffold_smiles(s) for s in unique_smiles}
    scaffolds.discard("__INVALID__")
    n_unique_scaffolds = len(scaffolds)
    diversity_ratio = float(n_unique_scaffolds / len(unique_smiles)) if unique_smiles else 0.0

    vals = df["pchembl_value"].to_numpy(dtype=float)
    n_lt6 = int(np.sum(vals < 6.0))

    endpoint_counts = {
        str(k): int(v)
        for k, v in df["standard_type"].value_counts().items()
    }
    subtype_breakdown = {
        st: int((df["target_subtype"] == st).sum())
        for st in SUBTYPES
    }

    per_subtype_stats = {}
    for st in SUBTYPES:
        sub_vals = df.loc[df["target_subtype"] == st, "pchembl_value"].to_numpy(dtype=float)
        per_subtype_stats[st] = {
            "n_records": int(len(sub_vals)),
            "pchembl_min": float(np.min(sub_vals)) if len(sub_vals) else None,
            "pchembl_max": float(np.max(sub_vals)) if len(sub_vals) else None,
            "pchembl_mean": float(np.mean(sub_vals)) if len(sub_vals) else None,
            "pchembl_median": float(np.median(sub_vals)) if len(sub_vals) else None,
            "n_weak_or_inactive_lt6": int(np.sum(sub_vals < 6.0)) if len(sub_vals) else 0,
        }

    return {
        "run_id": run_id,
        "assay_filter": "Ki and Kd only (pKi)",
        "n_raw_records": int(n_raw_records),
        "n_records_total": int(len(df)),
        "n_unique_molecules": int(len(unique_smiles)),
        "n_multi_subtype_molecules": n_multi_subtype,
        "n_replicates_dropped_conflict": n_replicates_dropped_conflict,
        "sources_breakdown": raw_sources,
        "scaffold_diversity": {
            "n_unique_scaffolds": int(n_unique_scaffolds),
            "diversity_ratio": round(diversity_ratio, 4),
        },
        "pchembl_stats": {
            "min": round(float(np.min(vals)), 4) if len(vals) else None,
            "max": round(float(np.max(vals)), 4) if len(vals) else None,
            "mean": round(float(np.mean(vals)), 4) if len(vals) else None,
            "median": round(float(np.median(vals)), 4) if len(vals) else None,
            "std": round(float(np.std(vals)), 4) if len(vals) else None,
            "n_weak_or_inactive_lt6": n_lt6,
            "pct_weak_or_inactive_lt6": round(float(100.0 * n_lt6 / len(vals)), 2) if len(vals) else 0.0,
        },
        "standard_type_breakdown": endpoint_counts,
        "target_subtype_breakdown": subtype_breakdown,
        "per_subtype": per_subtype_stats,
    }


def evaluate(
    mode: str = "precise",
    data_path: str = "data/raw",
    test_size: float = SCAFFOLD_TEST_SIZE,
    random_state: int = SCAFFOLD_SPLIT_SEED,
    out_path: Optional[str] = None,
    include_decoys: bool = False,
    run_id: Optional[str] = None,
) -> dict:
    """Evaluate saved XGBoost, LightGBM, RandomForest, and Consensus models on the held-out scaffold split."""
    if out_path is None:
        out_path = str(OUTPUTS_DIR / "evaluation_report.json")
    _run_id = run_id or RUN_ID

    df, _ = load_and_clean(
        data_path,
        save_lookup_path=None,
        mode=mode,
        include_decoys=include_decoys,
    )

    split_path = PROCESSED_DATA_DIR / "global_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    available_smiles = set(df["canonical_smiles"])
    train_smiles = set(split["train"]) & available_smiles
    test_smiles = set(split["test"]) & available_smiles

    train_df = df[df["canonical_smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["canonical_smiles"].isin(test_smiles)].reset_index(drop=True)

    y_train_all = train_df["pchembl_value"].to_numpy(dtype=float)
    y_test_all = test_df["pchembl_value"].to_numpy(dtype=float)

    xgb_models = _load_xgb_models()
    lgb_models = _load_lgb_models()
    rf_models = _load_rf_models()
    weights_map = _load_consensus_weights("precise")

    per_subtype: Dict[str, dict] = {}
    holdout_rows: List[pd.DataFrame] = []
    all_preds: List[np.ndarray] = []
    all_xgb_preds: List[np.ndarray] = []
    all_lgb_preds: List[np.ndarray] = []
    all_rf_preds: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    all_std: List[np.ndarray] = []
    all_lowers: List[np.ndarray] = []
    all_uppers: List[np.ndarray] = []

    for st in SUBTYPES:
        train_mask = (train_df["target_subtype"].values == st)
        test_mask = (test_df["target_subtype"].values == st)

        tr_sub_df = train_df.loc[train_mask].reset_index(drop=True)
        te_sub_df = test_df.loc[test_mask].reset_index(drop=True)
        ytr = y_train_all[train_mask]
        yte = y_test_all[test_mask]
        te_smiles = te_sub_df["canonical_smiles"].to_numpy()

        if len(yte) == 0:
            per_subtype[st] = {"n_test": 0, "skipped": True}
            continue

        st_pipeline = _load_scaler("precise", subtype=st)
        Xtr = _transform_df(tr_sub_df, st_pipeline)
        Xte = _transform_df(te_sub_df, st_pipeline)

        xgb_ens = xgb_models.get(st)
        lgb_ens = lgb_models.get(st)
        rf_ens = rf_models.get(st)
        if xgb_ens is None and lgb_ens is None and rf_ens is None:
            logger.warning("No production models found for %s", st)
            per_subtype[st] = {"n_test": int(len(yte)), "skipped": True, "error": "No model"}
            continue

        if xgb_ens is not None:
            p_xgb, std_xgb, low_xgb, high_xgb = _ensemble_predict(xgb_ens, Xte)
            p_xgb = np.atleast_1d(p_xgb).astype(float)
            std_xgb = np.atleast_1d(std_xgb).astype(float)
            low_xgb = np.atleast_1d(low_xgb).astype(float)
            high_xgb = np.atleast_1d(high_xgb).astype(float)
        else:
            p_xgb = std_xgb = low_xgb = high_xgb = None

        if lgb_ens is not None:
            p_lgb, std_lgb, low_lgb, high_lgb = _ensemble_predict(lgb_ens, Xte)
            p_lgb = np.atleast_1d(p_lgb).astype(float)
            std_lgb = np.atleast_1d(std_lgb).astype(float)
            low_lgb = np.atleast_1d(low_lgb).astype(float)
            high_lgb = np.atleast_1d(high_lgb).astype(float)
        else:
            p_lgb = std_lgb = low_lgb = high_lgb = None

        if rf_ens is not None:
            p_rf, std_rf, low_rf, high_rf = _ensemble_predict(rf_ens, Xte)
            p_rf = np.atleast_1d(p_rf).astype(float)
            std_rf = np.atleast_1d(std_rf).astype(float)
            low_rf = np.atleast_1d(low_rf).astype(float)
            high_rf = np.atleast_1d(high_rf).astype(float)
        else:
            p_rf = std_rf = low_rf = high_rf = None

        w_st = weights_map.get(st, {"XGBoost": 1 / 3, "LightGBM": 1 / 3, "RandomForest": 1 / 3})
        named_preds = [("XGBoost", p_xgb), ("LightGBM", p_lgb), ("RandomForest", p_rf)]
        avail_named = [(name, arr) for name, arr in named_preds if arr is not None]
        w_sum = sum(float(w_st.get(name, 1 / 3)) for name, _ in avail_named)
        if w_sum > 0:
            preds = sum(float(w_st.get(name, 1 / 3)) * arr for name, arr in avail_named) / w_sum
        else:
            preds = np.mean(np.vstack([arr for _, arr in avail_named]), axis=0)

        conformal_stds = [arr for arr in (std_xgb, std_lgb) if arr is not None]
        conformal_lows = [arr for arr in (low_xgb, low_lgb) if arr is not None]
        conformal_highs = [arr for arr in (high_xgb, high_lgb) if arr is not None]
        if conformal_lows:
            stds = np.mean(np.vstack(conformal_stds), axis=0)
            lowers = np.mean(np.vstack(conformal_lows), axis=0)
            uppers = np.mean(np.vstack(conformal_highs), axis=0)
        else:
            stds, lowers, uppers = std_rf, low_rf, high_rf

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(Xtr, ytr)
        base_preds = baseline.predict(Xte)

        coverage = float(np.mean((yte >= lowers) & (yte <= uppers)))
        mean_width = float(np.mean(uppers - lowers))
        mean_std = float(np.mean(stds))

        cons_mae = float(mean_absolute_error(yte, preds))
        cons_rmse = float(np.sqrt(mean_squared_error(yte, preds)))
        cons_r2 = float(r2_score(yte, preds))
        cons_rho = _safe_spearman(yte, preds)

        report_st = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "mae": cons_mae,
            "rmse": cons_rmse,
            "r2": cons_r2,
            "model_mae": cons_mae,
            "model_rmse": cons_rmse,
            "model_r2": cons_r2,
            "spearman_rho": cons_rho,
            "xgb_mae": float(mean_absolute_error(yte, p_xgb)) if p_xgb is not None else None,
            "xgb_rmse": float(np.sqrt(mean_squared_error(yte, p_xgb))) if p_xgb is not None else None,
            "xgb_r2": float(r2_score(yte, p_xgb)) if p_xgb is not None else None,
            "xgb_spearman": _safe_spearman(yte, p_xgb),
            "lgb_mae": float(mean_absolute_error(yte, p_lgb)) if p_lgb is not None else None,
            "lgb_rmse": float(np.sqrt(mean_squared_error(yte, p_lgb))) if p_lgb is not None else None,
            "lgb_r2": float(r2_score(yte, p_lgb)) if p_lgb is not None else None,
            "lgb_spearman": _safe_spearman(yte, p_lgb),
            "rf_mae": float(mean_absolute_error(yte, p_rf)) if p_rf is not None else None,
            "rf_rmse": float(np.sqrt(mean_squared_error(yte, p_rf))) if p_rf is not None else None,
            "rf_r2": float(r2_score(yte, p_rf)) if p_rf is not None else None,
            "rf_spearman": _safe_spearman(yte, p_rf),
            "baseline_mae": float(mean_absolute_error(yte, base_preds)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(yte, base_preds))),
            "baseline_r2": float(r2_score(yte, base_preds)),
            "baseline_spearman": _safe_spearman(yte, base_preds),
            "delta_mae": float(cons_mae - mean_absolute_error(yte, base_preds)),
            "uncertainty_mean_std": mean_std,
            "mean_interval_width": mean_width,
            "conformal_coverage_90": coverage,
            "calibration_quartiles": _calibration_quartiles(yte, preds, stds, lowers, uppers),
            "models": ["XGBoost", "LightGBM", "RandomForest", "Consensus"],
        }
        per_subtype[st] = report_st

        holdout_rows.append(
            pd.DataFrame(
                {
                    "smiles": te_smiles,
                    "subtype": st,
                    "y_true": yte,
                    "y_pred_xgb": p_xgb if p_xgb is not None else np.nan,
                    "y_pred_lgb": p_lgb if p_lgb is not None else np.nan,
                    "y_pred_rf": p_rf if p_rf is not None else np.nan,
                    "y_pred_consensus": preds,
                    "ci_low": lowers,
                    "ci_high": uppers,
                    "std_equiv": stds,
                }
            )
        )

        all_preds.append(preds)
        if p_xgb is not None:
            all_xgb_preds.append(p_xgb)
        if p_lgb is not None:
            all_lgb_preds.append(p_lgb)
        if p_rf is not None:
            all_rf_preds.append(p_rf)
        all_true.append(yte)
        all_std.append(stds)
        all_lowers.append(lowers)
        all_uppers.append(uppers)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_std = np.concatenate(all_std) if all_std else np.array([])
    y_lower = np.concatenate(all_lowers) if all_lowers else np.array([])
    y_upper = np.concatenate(all_uppers) if all_uppers else np.array([])
    y_xgb_all = np.concatenate(all_xgb_preds) if all_xgb_preds else np.array([])
    y_lgb_all = np.concatenate(all_lgb_preds) if all_lgb_preds else np.array([])
    y_rf_all = np.concatenate(all_rf_preds) if all_rf_preds else np.array([])

    base_all = DummyRegressor(strategy="mean")
    base_all.fit(np.zeros((len(y_train_all), 1)), y_train_all)
    base_pred_all = base_all.predict(np.zeros((len(y_true), 1)))

    overall_coverage = float(np.mean((y_true >= y_lower) & (y_true <= y_upper))) if len(y_true) else 0.0
    overall_width = float(np.mean(y_upper - y_lower)) if len(y_true) else 0.0
    overall_std = float(np.mean(y_std)) if len(y_std) else 0.0

    ov_mae = float(mean_absolute_error(y_true, y_pred)) if len(y_true) else None
    ov_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else None
    ov_r2 = float(r2_score(y_true, y_pred)) if len(y_true) else None
    ov_rho = _safe_spearman(y_true, y_pred) if len(y_true) else None

    has_xgb_all = len(y_xgb_all) == len(y_true) and len(y_true) > 0
    has_lgb_all = len(y_lgb_all) == len(y_true) and len(y_true) > 0
    has_rf_all = len(y_rf_all) == len(y_true) and len(y_true) > 0

    summary = {
        "run_id": _run_id,
        "mode": mode,
        "data_path": data_path,
        "assay_filter": "Ki and Kd only (pKi)",
        "include_decoys": False,
        "split": {"test_size": test_size, "random_state": random_state},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "models": ["XGBoost", "LightGBM", "RandomForest", "Consensus"],
        "overall": {
            "mae": ov_mae,
            "rmse": ov_rmse,
            "r2": ov_r2,
            "model_mae": ov_mae,
            "model_rmse": ov_rmse,
            "model_r2": ov_r2,
            "spearman_rho": ov_rho,
            "xgb_mae": float(mean_absolute_error(y_true, y_xgb_all)) if has_xgb_all else None,
            "xgb_rmse": float(np.sqrt(mean_squared_error(y_true, y_xgb_all))) if has_xgb_all else None,
            "xgb_r2": float(r2_score(y_true, y_xgb_all)) if has_xgb_all else None,
            "xgb_spearman": _safe_spearman(y_true, y_xgb_all) if has_xgb_all else None,
            "lgb_mae": float(mean_absolute_error(y_true, y_lgb_all)) if has_lgb_all else None,
            "lgb_rmse": float(np.sqrt(mean_squared_error(y_true, y_lgb_all))) if has_lgb_all else None,
            "lgb_r2": float(r2_score(y_true, y_lgb_all)) if has_lgb_all else None,
            "lgb_spearman": _safe_spearman(y_true, y_lgb_all) if has_lgb_all else None,
            "rf_mae": float(mean_absolute_error(y_true, y_rf_all)) if has_rf_all else None,
            "rf_rmse": float(np.sqrt(mean_squared_error(y_true, y_rf_all))) if has_rf_all else None,
            "rf_r2": float(r2_score(y_true, y_rf_all)) if has_rf_all else None,
            "rf_spearman": _safe_spearman(y_true, y_rf_all) if has_rf_all else None,
            "baseline_mae": float(mean_absolute_error(y_test_all, base_pred_all)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test_all, base_pred_all))),
            "baseline_r2": float(r2_score(y_test_all, base_pred_all)),
            "baseline_spearman": _safe_spearman(y_test_all, base_pred_all),
            "uncertainty_mean_std": overall_std,
            "mean_interval_width": overall_width,
            "conformal_coverage_90": overall_coverage,
            "calibration_quartiles": _calibration_quartiles(y_true, y_pred, y_std, y_lower, y_upper) if len(y_true) else [],
        },
        "per_subtype": per_subtype,
    }

    out_file = Path(out_path)
    save_with_run_id(summary, out_file.parent, out_file.stem, _run_id)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame(
        columns=[
            "smiles",
            "subtype",
            "y_true",
            "y_pred_xgb",
            "y_pred_lgb",
            "y_pred_rf",
            "y_pred_consensus",
            "ci_low",
            "ci_high",
            "std_equiv",
        ]
    )
    holdout_path = OUTPUTS_DIR / "holdout_predictions.csv"
    holdout_df.to_csv(holdout_path, index=False)
    logger.info("Saved holdout predictions (%d rows) to %s", len(holdout_df), holdout_path)

    diagnostics = _compute_dataset_diagnostics(df, data_path, _run_id)
    diag_path = OUTPUTS_DIR / "dataset_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Saved dataset diagnostics to %s", diag_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rep = evaluate(mode="precise")
    ov = rep["overall"]
    logger.info(
        "Evaluation complete: Consensus R2=%.4f, MAE=%.4f, Spearman=%.4f, Coverage90=%.2f%%",
        ov["model_r2"] or 0.0,
        ov["model_mae"] or 0.0,
        ov["spearman_rho"] or 0.0,
        (ov["conformal_coverage_90"] or 0.0) * 100.0,
    )
