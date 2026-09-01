import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_and_clean
from src.features import build_feature_matrix
from src.predictor import _load_xgb_models, _load_rf_models, _load_lgb_models, _ensemble_predict, SUBTYPES, _try_gnn_predict
from src.config import (
    SUBTYPES, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
    CONFORMAL_ALPHA, LOG_LEVEL, RUN_ID,
)
def save_with_run_id(data: dict, out_dir_or_path: str, filename_or_runid: str, run_id: str = None) -> None:
    """Save data to versioned JSON path and latest JSON path."""
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
    
    with open(versioned, "w") as f:
        json.dump(save_data, f, indent=2)
    with open(target_file, "w") as f:
        json.dump(save_data, f, indent=2)

logger = logging.getLogger(__name__)


def _calibration_quartiles(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> List[dict]:
    """
    Group predictions into quartiles by predicted uncertainty.

    A well-calibrated model shows monotonically increasing MAE
    as predicted uncertainty increases. This is a key diagnostic
    for conformal prediction quality.
    """
    n = len(y_true)
    if n < 8:
        return []

    order = np.argsort(y_std)
    bins = np.array_split(order, 4)

    out = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        mae = float(np.mean(np.abs(y_true[idx] - y_pred[idx])))
        out.append({
            "bin": i,
            "n": int(len(idx)),
            "std_mean": float(np.mean(y_std[idx])),
            "mae_mean": mae,
        })
    return out


def evaluate(mode: str = "precise",
             data_path: str = "data/raw",
             test_size: float = 0.2,
             random_state: int = 42,
             out_path: str | None = None,
             include_decoys: bool = True,
             run_id: str | None = None) -> dict:
    """
    Evaluate models with real conformal prediction intervals.

    CRITICAL FIX:
    - The old code had std_mean=0.0 because retrain_production.py saved
      raw XGBRegressors without MAPIE conformal wrapping.
    - Now retrain_production.py wraps models with CrossConformalRegressor,
      so predict_interval() returns real uncertainty estimates.
    - The evaluator still gracefully handles legacy raw models (std=0 fallback).
    """
    if out_path is None:
        out_path = str(OUTPUTS_DIR / "validoutput" / mode / f"evaluation_{mode}_report.json")

    df, _ = load_and_clean(data_path, mode=mode, include_decoys=include_decoys)

    split_path = PROCESSED_DATA_DIR / "global_split.json"
    with open(split_path) as f:
        split = json.load(f)
    train_smiles = set(split["train"])
    test_smiles = set(split["test"])

    available_smiles = set(df["canonical_smiles"])
    train_smiles = train_smiles & available_smiles
    test_smiles = test_smiles & available_smiles

    train_df = df[df["canonical_smiles"].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df["canonical_smiles"].isin(test_smiles)].reset_index(drop=True)

    import pickle
    from src.features import _morgan_bits, _maccs_bits, _all_descriptors

    scaler_path = MODELS_DIR / mode / f"scaler_{mode}.pkl"
    if not scaler_path.exists():
        scaler_path = MODELS_DIR / "scaler.pkl"

    logger.info("Loading production feature pipeline from %s", scaler_path)
    with open(scaler_path, "rb") as f:
        pipeline = pickle.load(f)

    def transform_df(df):
        from joblib import Parallel, delayed
        smiles = df["canonical_smiles"].tolist()
        Xfp = np.vstack(Parallel(n_jobs=-1)(delayed(_morgan_bits)(s) for s in smiles))
        Xmaccs = np.vstack(Parallel(n_jobs=-1)(delayed(_maccs_bits)(s) for s in smiles))
        Xdesc = np.vstack(Parallel(n_jobs=-1)(delayed(_all_descriptors)(s) for s in smiles))
        Xdesc_s = pipeline.transform(Xdesc)
        return np.hstack([Xfp, Xmaccs, Xdesc_s]).astype(np.float32)

    X_train = transform_df(train_df)
    X_test = transform_df(test_df)

    y_train_all = train_df["pchembl_value"].to_numpy(dtype=float)
    y_test_all = test_df["pchembl_value"].to_numpy(dtype=float)

    models = _load_xgb_models()
    rf_models = _load_rf_models()
    lgb_models = _load_lgb_models()

    per_subtype: Dict[str, dict] = {}
    all_preds = []
    all_true = []
    all_std = []
    all_lowers = []
    all_uppers = []

    for st in SUBTYPES:
        train_mask = (train_df["target_subtype"].values == st)
        test_mask = (test_df["target_subtype"].values == st)

        Xtr = X_train[train_mask]
        ytr = y_train_all[train_mask]
        Xte = X_test[test_mask]
        yte = y_test_all[test_mask]

        if len(yte) == 0:
            per_subtype[st] = {"n_test": 0, "skipped": True}
            continue

        model_ens = models.get(st)
        if model_ens is None:
            logger.warning("No model found for %s", st)
            per_subtype[st] = {"n_test": int(len(yte)), "skipped": True, "error": "No model"}
            continue

        # Use vectorized prediction directly on the batch Xte
        preds, stds, lowers, uppers = _ensemble_predict(model_ens, Xte)
        preds = np.atleast_1d(preds)
        stds = np.atleast_1d(stds)
        lowers = np.atleast_1d(lowers)
        uppers = np.atleast_1d(uppers)

        mean_std = float(np.mean(stds))
        if mean_std < 1e-6:
            logger.warning("%s: Conformal uncertainty near-zero (%.6f). Model type: %s",
                           st, mean_std, type(model_ens).__name__)
        else:
            logger.info("%s: Mean conformal uncertainty = %.4f (model type: %s)",
                        st, mean_std, type(model_ens).__name__)

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(Xtr, ytr)
        base_preds = baseline.predict(Xte)

        in_interval = np.sum((yte >= lowers) & (yte <= uppers))
        coverage = float(in_interval / len(yte)) if len(yte) > 0 else 0.0

        report_st = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "model_mae": float(mean_absolute_error(yte, preds)),
            "model_rmse": float(np.sqrt(mean_squared_error(yte, preds))),
            "model_r2": float(r2_score(yte, preds)),
            "baseline_mae": float(mean_absolute_error(yte, base_preds)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(yte, base_preds))),
            "baseline_r2": float(r2_score(yte, base_preds)),
            "delta_mae": float(mean_absolute_error(yte, preds) - mean_absolute_error(yte, base_preds)),
            "uncertainty_mean_std": mean_std,
            "conformal_coverage_90": coverage,
            "calibration_quartiles": _calibration_quartiles(yte, preds, stds),
            "model_type": type(model_ens).__name__,
        }

        try:
            from src.gnn_model import MoleculeGNN, smiles_to_graph
            import torch
            from torch_geometric.loader import DataLoader

            gnn_model_path = MODELS_DIR / "gnn" / f"gnn_{st.lower()}_model.pt"
            if gnn_model_path.exists():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Safe to use weights_only=False here as checkpoints are generated locally by our pipeline.
                checkpoint = torch.load(gnn_model_path, map_location=device, weights_only=False)
                gnn_model = MoleculeGNN(
                    node_dim=checkpoint.get("node_dim", 140),
                    edge_dim=checkpoint.get("edge_dim", 7),
                    hidden_dim=checkpoint.get("hidden_dim", 256),
                    num_layers=checkpoint.get("num_layers", 3),
                ).to(device)
                gnn_model.load_state_dict(checkpoint["model_state_dict"])
                gnn_model.eval()

                test_smiles_st = test_df.loc[test_mask, "canonical_smiles"].tolist()
                test_graphs = []
                for idx, smi in enumerate(test_smiles_st):
                    g = smiles_to_graph(smi)
                    if g is not None:
                        g.y = torch.tensor([yte[idx]], dtype=torch.float)
                        test_graphs.append(g)

                if test_graphs:
                    loader = DataLoader(test_graphs, batch_size=128, shuffle=False)
                    preds_gnn = []
                    true_gnn = []
                    with torch.no_grad():
                        for data in loader:
                            data = data.to(device)
                            out = gnn_model(data)
                            preds_gnn.extend(out.cpu().numpy().tolist())
                            true_gnn.extend(data.y.cpu().numpy().tolist())

                    report_st["gnn_mae"] = float(mean_absolute_error(true_gnn, preds_gnn))
                    report_st["gnn_rmse"] = float(np.sqrt(mean_squared_error(true_gnn, preds_gnn)))
                    report_st["gnn_r2"] = float(r2_score(true_gnn, preds_gnn))
                    logger.info("  GNN %s: MAE=%.4f, R2=%.4f", st, report_st["gnn_mae"], report_st["gnn_r2"])
                else:
                    report_st["gnn_mae"] = None
                    report_st["gnn_rmse"] = None
                    report_st["gnn_r2"] = None
            else:
                report_st["gnn_mae"] = None
                report_st["gnn_rmse"] = None
                report_st["gnn_r2"] = None
        except Exception as e:
            logger.warning("GNN evaluation failed for %s: %s", st, e)
            report_st["gnn_mae"] = None
            report_st["gnn_rmse"] = None
            report_st["gnn_r2"] = None

        rf_ens = rf_models.get(st)
        if rf_ens:
            rf_preds = rf_ens.predict(Xte)
            report_st["rf_mae"] = float(mean_absolute_error(yte, rf_preds))
            report_st["rf_rmse"] = float(np.sqrt(mean_squared_error(yte, rf_preds)))
            report_st["rf_r2"] = float(r2_score(yte, rf_preds))
        else:
            report_st["rf_mae"] = None
            report_st["rf_rmse"] = None
            report_st["rf_r2"] = None

        lgb_ens = lgb_models.get(st)
        if lgb_ens:
            lgb_preds = lgb_ens.predict(Xte)
            report_st["lgb_mae"] = float(mean_absolute_error(yte, lgb_preds))
            report_st["lgb_rmse"] = float(np.sqrt(mean_squared_error(yte, lgb_preds)))
            report_st["lgb_r2"] = float(r2_score(yte, lgb_preds))
            logger.info("  LightGBM %s: MAE=%.4f, R²=%.4f", st, report_st["lgb_mae"], report_st["lgb_r2"])
        else:
            report_st["lgb_mae"] = None
            report_st["lgb_rmse"] = None
            report_st["lgb_r2"] = None

        per_subtype[st] = report_st

        all_preds.append(preds)
        all_true.append(yte)
        all_std.append(stds)
        all_lowers.append(lowers)
        all_uppers.append(uppers)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_std = np.concatenate(all_std) if all_std else np.array([])

    base_all = DummyRegressor(strategy="mean")
    base_all.fit(X_train, y_train_all)
    base_pred_all = base_all.predict(X_test)

    overall_coverage = 0.0
    if len(y_true) > 0 and len(all_lowers) > 0:
        y_lower = np.concatenate(all_lowers)
        y_upper = np.concatenate(all_uppers)
        overall_coverage = float(np.sum((y_true >= y_lower) & (y_true <= y_upper)) / len(y_true))

    _run_id = run_id or RUN_ID

    summary = {
        "run_id": _run_id,
        "mode": mode,
        "data_path": data_path,
        "include_decoys": include_decoys,
        "split": {"test_size": test_size, "random_state": random_state},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "overall": {
            "model_mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else None,
            "model_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else None,
            "model_r2": float(r2_score(y_true, y_pred)) if len(y_true) else None,
            "baseline_mae": float(mean_absolute_error(y_test_all, base_pred_all)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test_all, base_pred_all))),
            "baseline_r2": float(r2_score(y_test_all, base_pred_all)),
            "conformal_coverage_90": overall_coverage,
            "calibration_quartiles": _calibration_quartiles(y_true, y_pred, y_std) if len(y_true) else [],
        },
        "per_subtype": per_subtype,
    }

    save_with_run_id(summary, Path(out_path).parent if out_path else OUTPUTS_DIR,
                     Path(out_path).stem if out_path else "evaluation",
                     _run_id)
    logger.info("Evaluation report written [run_id=%s]", _run_id)

    if overall_coverage > 0:
        logger.info("Overall 90%% conformal coverage: %.2f%%", overall_coverage * 100)
        for st, r in per_subtype.items():
            if "uncertainty_mean_std" in r and r["uncertainty_mean_std"] > 0:
                logger.info("  %s: std=%.4f, coverage=%.2f%%",
                            st, r["uncertainty_mean_std"], r.get("conformal_coverage_90", 0) * 100)

    return summary


def evaluate_actives_only(mode: str = "precise",
                          data_path: str = "data/raw",
                          run_id: str | None = None) -> dict:
    """
    Separate evaluation on actives-only (no decoys) for honest reporting.
    This prevents inflated R² from easy-to-predict decoy compounds.
    """
    _run_id = run_id or RUN_ID
    out_path = str(OUTPUTS_DIR / "validoutput" / mode / f"evaluation_{mode}_actives_only")
    return evaluate(
        mode=mode,
        data_path=data_path,
        out_path=out_path,
        include_decoys=False,
        run_id=_run_id,
    )


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("=" * 60)
    logger.info("EVALUATING WITH DECOYS (full dataset)")
    logger.info("=" * 60)
    rep_full = evaluate(mode="precise")

    logger.info("=" * 60)
    logger.info("EVALUATING ACTIVES-ONLY (honest baseline)")
    logger.info("=" * 60)
    rep_actives = evaluate_actives_only(mode="precise")

    if rep_full["overall"]["model_r2"] is not None and rep_actives["overall"]["model_r2"] is not None:
        logger.info("Full dataset R² = %.4f", rep_full['overall']['model_r2'])
        logger.info("Actives-only R² = %.4f", rep_actives['overall']['model_r2'])
