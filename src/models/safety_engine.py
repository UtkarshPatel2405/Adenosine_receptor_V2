"""Pillar 3: Adenosine Cardiac Safety & Anti-Target Off-Target Panel."""
from typing import Dict, Any
from rdkit import Chem

_XANTHINE_CORE = Chem.MolFromSmarts("c1nc2c(=O)[nH]c(=O)n(C)c2n1")
_THEOPHYLLINE_CORE = Chem.MolFromSmarts("Cn1c(=O)c2[nH]cnc2n(C)c1=O")


def evaluate_safety_profile(predictions: Dict[str, float], smiles: str, in_domain: bool = True) -> Dict[str, Any]:
    """Assess A1 cardiac AV block liability, A3 mast cell risk, and PDE off-target cross-reactivity."""
    if not in_domain:
        return {
            "a1_bradycardia_risk": "Low (Out of Domain)",
            "a3_mast_cell_risk": "Low (Out of Domain)",
            "pde_cross_reactivity": "Low (Out of Domain)",
            "overall_safety_index": "N/A (Extrapolation)",
        }

    p_a1 = float(predictions.get("A1", 0.0) or 0.0)
    p_a2a = float(predictions.get("A2A", 0.0) or 0.0)
    p_a3 = float(predictions.get("A3", 0.0) or 0.0)

    # 1. A1 Bradycardia / AV Block Liability (Critical when A1 pChEMBL >= 7.0 and not heavily selective elsewhere)
    if p_a1 >= 7.5:
        a1_risk = "HIGH (Sub-nanomolar A1 Potency - AV Block Liability)"
    elif p_a1 >= 6.0:
        a1_risk = "MODERATE (Micromolar A1 Activity - Monitor Heart Rate)"
    else:
        a1_risk = "LOW (A1 pChEMBL < 6.0 - Minimal Cardiac AV Risk)"

    # 2. A3 Mast Cell Degranulation & Bronchoconstriction
    if p_a3 >= 7.5:
        a3_risk = "HIGH (Potent A3 Agonism - Mast Cell Degranulation Risk)"
    elif p_a3 >= 6.0:
        a3_risk = "MODERATE (Sub-micromolar A3 Activity)"
    else:
        a3_risk = "LOW (Minimal A3 Mast Cell Liability)"

    # 3. Phosphodiesterase (PDE1-10) Off-Target Cross-Reactivity
    mol = Chem.MolFromSmiles(smiles)
    has_xanthine = mol.HasSubstructMatch(_XANTHINE_CORE) or mol.HasSubstructMatch(_THEOPHYLLINE_CORE) if mol else False
    _PURINE_LIKE = Chem.MolFromSmarts("c1nc2ncnc2n1")
    has_purine_like = mol.HasSubstructMatch(_PURINE_LIKE) if mol else False
    if has_xanthine:
        pde_risk = "HIGH (Xanthine Core Scaffold - Known PDE1-10 Cross-Inhibition)"
    elif has_purine_like:
        pde_risk = "MODERATE (Purine-like Heterocycle - Weak PDE Affinity Possible)"
    else:
        pde_risk = "LOW (Non-xanthine Chemotype - Selective against PDEs)"

    # Overall Safety Score
    risk_count = sum(1 for r in [a1_risk, a3_risk, pde_risk] if "HIGH" in r)
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
