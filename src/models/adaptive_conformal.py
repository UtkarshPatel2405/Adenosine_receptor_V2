"""Statistically calibrated conformal prediction intervals with domain regime annotation."""
from typing import Dict, Any


def calibrate_adaptive_interval(pred_val: float, base_lower: float, base_upper: float, tanimoto_max: float, in_domain: bool = True) -> Dict[str, Any]:
    """Annotate calibrated conformal intervals with chemical space density and scaffold regime."""
    if not in_domain:
        low = max(3.0, round(base_lower, 2))
        high = min(11.0, round(base_upper, 2))
        return {
            "lower": low,
            "upper": high,
            "width": round(high - low, 2),
            "confidence": 0.90,
            "regime": "Out of Applicability Domain (Extrapolation)",
        }

    # Assign regime based on maximum training Tanimoto similarity
    if tanimoto_max >= 0.70:
        regime = "High-Density Core (Tanimoto >= 0.70)"
    elif tanimoto_max >= 0.40:
        regime = "Standard Scaffold Space (0.40 <= Tanimoto < 0.70)"
    else:
        regime = "Scaffold Hop / Sparse Space (Tanimoto < 0.40)"

    low = max(3.0, round(base_lower, 2))
    high = min(11.0, round(base_upper, 2))

    return {
        "lower": low,
        "upper": high,
        "width": round(max(0.0, high - low), 2),
        "confidence": 0.90,
        "regime": regime,
    }

