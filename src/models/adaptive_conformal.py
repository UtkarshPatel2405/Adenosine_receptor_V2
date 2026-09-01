"""Silent Bug 4: Locally Adaptive / Scaffold-Conditioned Conformal Uncertainty."""
from typing import Dict, Any


def calibrate_adaptive_interval(pred_val: float, base_lower: float, base_upper: float, tanimoto_max: float, in_domain: bool = True) -> Dict[str, float]:
    """Dynamically adjust conformal intervals based on chemical space density & scaffold distance."""
    if not in_domain:
        # Out-of-domain uncalibrated extrapolation: maximal uncertainty interval
        low = max(3.0, round(pred_val - 2.5, 2))
        high = min(11.0, round(pred_val + 2.5, 2))
        return {"lower": low, "upper": high, "width": round(high - low, 2), "confidence": 0.50, "regime": "Extrapolation"}

    # Base half-width from MAPIE
    raw_half_width = max(0.35, (base_upper - base_lower) / 2.0)

    # Heteroscedastic scaling factor based on topological proximity to reference training chemotypes
    if tanimoto_max >= 0.70:
        # Heavily explored core (e.g. adenine/ribose analogs): low heteroscedastic noise
        scale_factor = 0.75
        regime = "High-Density Core"
    elif tanimoto_max >= 0.40:
        # Standard chemical space coverage
        scale_factor = 1.00
        regime = "Standard Scaffold Space"
    else:
        # Borderline scaffold hop (0.25 <= tanimoto_max < 0.40): wider conformal intervals
        scale_factor = 1.60
        regime = "Scaffold Hop / Rare Chemotype"

    adaptive_half = raw_half_width * scale_factor
    low = max(3.0, round(pred_val - adaptive_half, 2))
    high = min(11.0, round(pred_val + adaptive_half, 2))

    return {
        "lower": low,
        "upper": high,
        "width": round(high - low, 2),
        "confidence": 0.90,
        "regime": regime,
    }
