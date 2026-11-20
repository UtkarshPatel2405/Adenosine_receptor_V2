"""Deterministic SMARTS-based adenosine target liability and PDE off-target hypotheses."""
from typing import Any, Dict

from rdkit import Chem

_XANTHINE_CORE = Chem.MolFromSmarts("O=c1[#7;r6]c(=O)c2[#7;r5]c[#7;r5]c2[#7;r6]1")
_PURINE_LIKE = Chem.MolFromSmarts("n1cnc2[#7;r5]c[#7;r5]c2c1")


def evaluate_safety_profile(
    predictions: Dict[str, float],
    smiles: str,
    in_domain: bool = True,
) -> Dict[str, Any]:
    """Screen deterministic A1/A3 affinity liabilities and xanthine/purine PDE structural alerts."""
    if not in_domain:
        return {
            "a1_bradycardia_risk": "Low (Out of Domain)",
            "a3_mast_cell_risk": "Low (Out of Domain)",
            "pde_cross_reactivity": "Low (Out of Domain)",
            "overall_safety_index": "N/A (Extrapolation)",
        }

    p_a1 = float(predictions.get("A1", 0.0) or 0.0)
    p_a3 = float(predictions.get("A3", 0.0) or 0.0)

    if p_a1 >= 7.5:
        a1_risk = "HIGH (Sub-nanomolar A1 Potency - AV Block Liability)"
    elif p_a1 >= 6.0:
        a1_risk = "MODERATE (Micromolar A1 Activity - Monitor Heart Rate)"
    else:
        a1_risk = "LOW (A1 pChEMBL < 6.0 - Minimal Cardiac AV Risk)"

    if p_a3 >= 7.5:
        a3_risk = "HIGH (Potent A3 Agonism - Mast Cell Degranulation Risk)"
    elif p_a3 >= 6.0:
        a3_risk = "MODERATE (Sub-micromolar A3 Activity)"
    else:
        a3_risk = "LOW (Minimal A3 Mast Cell Liability)"

    mol = Chem.MolFromSmiles(smiles)
    has_xanthine = bool(mol and mol.HasSubstructMatch(_XANTHINE_CORE))
    has_purine_like = bool(mol and mol.HasSubstructMatch(_PURINE_LIKE))

    if has_xanthine:
        pde_risk = "HIGH (Xanthine Core Scaffold - Known PDE1-10 Cross-Inhibition)"
    elif has_purine_like:
        pde_risk = "MODERATE (Purine-like Heterocycle - Weak PDE Affinity Possible)"
    else:
        pde_risk = "LOW (Non-xanthine Chemotype - Selective against PDEs)"

    risk_count = sum(1 for r in (a1_risk, a3_risk, pde_risk) if "HIGH" in r)
    if risk_count == 0:
        safety_idx = "FAVORABLE (Zero High-Risk Target Liabilities)"
    elif risk_count == 1:
        safety_idx = "MONITORED (1 High-Risk Liability Identified)"
    else:
        safety_idx = "UNFAVORABLE (Multiple Off-Target / Cardiac Liabilities)"

    return {
        "a1_bradycardia_risk": a1_risk,
        "a3_mast_cell_risk": a3_risk,
        "pde_cross_reactivity": pde_risk,
        "overall_safety_index": safety_idx,
    }
