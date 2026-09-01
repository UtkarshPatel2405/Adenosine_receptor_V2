"""Pharmacological selectivity spectrum & pairwise differential calculation engine."""
from typing import Dict, Any, List, Tuple

SUBTYPE_PAIRS = [
    ("A1", "A2A"),
    ("A1", "A2B"),
    ("A1", "A3"),
    ("A2A", "A2B"),
    ("A2A", "A3"),
    ("A2B", "A3"),
]


def compute_selectivity_spectrum(predictions: Dict[str, float], canon_smiles: str = "", in_domain: bool = True) -> Dict[str, Any]:
    """Calculate multi-target selectivity profile, rank hierarchy, and pairwise differentials."""
    subtypes = ["A1", "A2A", "A2B", "A3"]
    valid_scores = {s: float(predictions[s]) for s in subtypes if s in predictions and predictions[s] is not None}

    # Rank-order subtypes by predicted affinity
    ranked = sorted(valid_scores.keys(), key=lambda s: valid_scores[s], reverse=True)
    best_target = ranked[0] if ranked else "N/A"
    top_score = valid_scores[ranked[0]] if ranked else 0.0

    # Domain and activity boundary validation
    if not in_domain:
        margin = 0.0
        margin_fold = 1.0
        classification = "Outside Applicability Domain (Extrapolation / Non-Adenosine Scaffold)"
    elif top_score < 5.0:
        margin = 0.0
        margin_fold = 1.0
        classification = "Inactive / Sub-micromolar Non-binder (pChEMBL < 5.0)"
    elif len(ranked) >= 2:
        runner_up = valid_scores[ranked[1]]
        margin = max(0.0, top_score - runner_up)
        margin_fold = 10 ** margin
        if margin >= 2.0:
            classification = f"Human {ranked[0]} Highly Selective ({margin_fold:.1f}x margin)"
        elif margin >= 1.0:
            classification = f"Human {ranked[0]} Subtype-Selective ({margin_fold:.1f}x margin)"
        elif margin >= 0.5:
            classification = f"Human {ranked[0]} Preferring ({margin_fold:.1f}x margin)"
        else:
            classification = f"Pan-Adenosine / Multi-Target ({margin_fold:.1f}x delta)"
    else:
        margin = 0.0
        margin_fold = 1.0
        classification = "Equipotent Profile"

    # Complete pairwise differentials
    pairwise: Dict[str, float] = {}
    for subA, subB in SUBTYPE_PAIRS:
        valA = valid_scores.get(subA)
        valB = valid_scores.get(subB)
        if valA is not None and valB is not None:
            pairwise[f"{subA}_vs_{subB}"] = round(valA - valB, 3)

    return {
        "best_target": best_target,
        "rank_hierarchy": ranked,
        "selectivity_margin": round(margin, 3),
        "selectivity_fold": round(margin_fold, 2),
        "classification": classification,
        "pairwise_deltas": pairwise,
    }
