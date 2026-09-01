"""Inference engine executing base models, conformal prediction, and stacked blending."""
from typing import Tuple, Dict, Any, Optional
import numpy as np

_ZERO_RESULT: Tuple[float, float, Dict[str, float]] = (
    0.0, 0.0, {"lower": 0.0, "upper": 0.0, "width": 0.0}
)


def _ensemble_predict(models, x: np.ndarray):
    """Execute prediction on a model or model ensemble and compute uncertainty bounds."""
    if not models or x is None:
        return 0.0, 0.0, 0.0, 0.0

    primary = models[0] if isinstance(models, list) else models
    mtype = type(primary).__name__
    is_1d = isinstance(x, np.ndarray) and x.ndim == 1
    x_2d = x.reshape(1, -1) if is_1d else np.asarray(x)

    if mtype in ("CrossConformalRegressor", "MapieRegressor"):
        try:
            y_pred, y_pis = primary.predict(x_2d, alpha=0.10)
            if is_1d:
                mean_val = float(y_pred[0])
                low = float(y_pis[0, 0, 0])
                high = float(y_pis[0, 1, 0])
                std_equiv = max(0.0, (high - low) / 3.29)
                return mean_val, std_equiv, low, high
            else:
                mean_arr = np.asarray(y_pred).ravel()
                low_arr = np.asarray(y_pis[:, 0, 0]).ravel()
                high_arr = np.asarray(y_pis[:, 1, 0]).ravel()
                std_arr = np.maximum(0.0, (high_arr - low_arr) / 3.29)
                return mean_arr, std_arr, low_arr, high_arr
        except Exception:
            pass

    if isinstance(models, list):
        preds = []
        for m in models:
            try:
                preds.append(m.predict(x_2d))
            except Exception:
                pass
        if preds:
            p_mat = np.array(preds)
            mean_val = np.mean(p_mat, axis=0)
            std_val = np.std(p_mat, axis=0)
            if is_1d:
                m = float(mean_val[0])
                s = float(std_val[0])
                return m, s, m - 1.645 * s, m + 1.645 * s
            return mean_val, std_val, mean_val - 1.645 * std_val, mean_val + 1.645 * std_val

    try:
        pred = primary.predict(x_2d)
        if is_1d:
            val = float(pred[0])
            return val, 0.0, val - 0.5, val + 0.5
        val_arr = np.asarray(pred).ravel()
        return val_arr, np.zeros_like(val_arr), val_arr - 0.5, val_arr + 0.5
    except Exception:
        return (0.0, 0.0, 0.0, 0.0) if is_1d else (None, None, None, None)


def _predict_one_subtype(model_dict: Dict[str, Any], x: Optional[np.ndarray], st: str) -> Tuple[float, float, Dict[str, float]]:
    """Predict for a specific subtype using the provided model dictionary."""
    if st in model_dict and x is not None:
        m, s, low, high = _ensemble_predict(model_dict[st], x)
        return float(m), float(s), {"lower": round(float(low), 3), "upper": round(float(high), 3), "width": round(float(high - low), 3)}
    return _ZERO_RESULT


def _try_gnn_predict(smiles: str, subtype: str) -> Optional[float]:
    """Graceful GNN / MPNN graph model evaluation fallback."""
    try:
        from src.gnn_model import predict_single_smiles
        return predict_single_smiles(smiles, subtype)
    except Exception:
        return None
