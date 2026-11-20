"""Inference engine executing conformal XGBoost and LightGBM models."""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.config import CONFORMAL_ALPHA

logger = logging.getLogger(__name__)

_ZERO_RESULT: Tuple[float, float, Dict[str, float]] = (
    0.0,
    0.0,
    {"lower": 0.0, "upper": 0.0, "width": 0.0},
)


def _ensemble_predict(models, x: np.ndarray):
    """Execute point prediction and 90% conformal uncertainty bounds on a model or ensemble."""
    if not models or x is None:
        return 0.0, 0.0, 0.0, 0.0

    primary = models[0] if isinstance(models, list) else models
    mtype = type(primary).__name__
    is_1d = isinstance(x, np.ndarray) and x.ndim == 1
    x_2d = x.reshape(1, -1) if is_1d else np.asarray(x)

    if mtype in ("CrossConformalRegressor", "MapieRegressor"):
        try:
            if hasattr(primary, "predict_interval"):
                y_pred, y_pis = primary.predict_interval(x_2d)
            else:
                y_pred, y_pis = primary.predict(x_2d, alpha=CONFORMAL_ALPHA)

            mean_arr = np.asarray(y_pred, dtype=float).ravel()
            raw_low = np.asarray(y_pis[:, 0, 0], dtype=float).ravel()
            raw_high = np.asarray(y_pis[:, 1, 0], dtype=float).ravel()
            low_arr = np.minimum(raw_low, raw_high)
            high_arr = np.maximum(raw_low, raw_high)
            std_arr = np.maximum(0.0, (high_arr - low_arr) / 3.29)

            if is_1d:
                return float(mean_arr[0]), float(std_arr[0]), float(low_arr[0]), float(high_arr[0])
            return mean_arr, std_arr, low_arr, high_arr
        except Exception as exc:
            logger.warning("Conformal interval prediction failed for %s: %s", mtype, exc)

    if isinstance(primary, dict) and "estimator" in primary and "q_half_width" in primary:
        try:
            est = primary["estimator"]
            q_hw = float(primary["q_half_width"])
            mean_arr = np.asarray(est.predict(x_2d), dtype=float).ravel()
            low_arr = mean_arr - q_hw
            high_arr = mean_arr + q_hw
            std_arr = np.full_like(mean_arr, max(0.0, (2.0 * q_hw) / 3.29))
            if is_1d:
                return float(mean_arr[0]), float(std_arr[0]), float(low_arr[0]), float(high_arr[0])
            return mean_arr, std_arr, low_arr, high_arr
        except Exception as exc:
            logger.warning("OOB conformal RF prediction failed: %s", exc)

    if isinstance(models, list):
        preds = []
        for m in models:
            try:
                preds.append(np.asarray(m.predict(x_2d), dtype=float).ravel())
            except Exception as exc:
                logger.warning("Ensemble member %s prediction failed: %s", type(m).__name__, exc)
        if preds:
            p_mat = np.vstack(preds)
            mean_val = np.mean(p_mat, axis=0)
            std_val = np.std(p_mat, axis=0)
            low_val = mean_val - 1.645 * std_val
            high_val = mean_val + 1.645 * std_val
            if is_1d:
                return float(mean_val[0]), float(std_val[0]), float(low_val[0]), float(high_val[0])
            return mean_val, std_val, low_val, high_val

    try:
        pred = np.asarray(primary.predict(x_2d), dtype=float).ravel()
        if is_1d:
            val = float(pred[0])
            return val, 0.0, val - 0.5, val + 0.5
        return pred, np.zeros_like(pred), pred - 0.5, pred + 0.5
    except Exception as exc:
        logger.warning("Base model prediction failed for %s: %s", mtype, exc)
        return (0.0, 0.0, 0.0, 0.0) if is_1d else (None, None, None, None)


def _predict_one_subtype(
    model_dict: Dict[str, Any],
    x: Optional[np.ndarray],
    st: str,
) -> Tuple[float, float, Dict[str, float]]:
    """Predict pChEMBL and raw conformal bounds for one subtype."""
    if st in model_dict and x is not None:
        m, s, low, high = _ensemble_predict(model_dict[st], x)
        low_f = min(float(low), float(high))
        high_f = max(float(low), float(high))
        return (
            float(m),
            float(s),
            {
                "lower": round(low_f, 3),
                "upper": round(high_f, 3),
                "width": round(max(0.0, high_f - low_f), 3),
            },
        )
    return _ZERO_RESULT
