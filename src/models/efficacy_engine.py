"""Deterministic 2D SMARTS pharmacophore heuristics for functional efficacy mode-of-action."""
from typing import Any, Dict

from rdkit import Chem

_RIBOSE_SMARTS = Chem.MolFromSmarts("C1OC(CO)C(O)C1O")
_CARBOXAMIDE_RIBOSE = Chem.MolFromSmarts("C1OC(C(=O)N)C(O)C1O")
_XANTHINE_CORE = Chem.MolFromSmarts("O=c1[#7;r6]c(=O)c2[#7;r5]c[#7;r5]c2[#7;r6]1")
_TRIAZOLOTRIAZINE = Chem.MolFromSmarts("c1nc2ncnn2cn1")
_TRIAZOLOPYRIMIDINE = Chem.MolFromSmarts("c1cc2ncnn2cn1")
_QUINAZOLINE_UREA = Chem.MolFromSmarts("c1ccc2ncnc(NC(=O)N)c2c1")


def predict_functional_efficacy(
    smiles: str,
    primary_subtype: str,
    max_pchembl: float,
    in_domain: bool = True,
    threshold: float = 5.0,
) -> Dict[str, Any]:
    """Assign deterministic 2D SMARTS MoA structural hypotheses (heuristic, not an ML probability)."""
    cutoff = max(5.0, float(threshold))
    if not in_domain or max_pchembl < cutoff or primary_subtype not in ("A1", "A2A", "A2B", "A3"):
        return {
            "mode_of_action": "Inactive / Non-binder",
            "efficacy_class": "Non-functional",
            "activation_probability": 0.0,
            "signaling_pathway": "None (Predicted Inactive)",
            "therapeutic_indication": "None (Predicted Inactive)",
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mode_of_action": "Unknown",
            "efficacy_class": "Unknown",
            "activation_probability": 0.0,
            "signaling_pathway": "None",
            "therapeutic_indication": "N/A",
        }

    has_carboxamide = mol.HasSubstructMatch(_CARBOXAMIDE_RIBOSE)
    has_ribose = has_carboxamide or mol.HasSubstructMatch(_RIBOSE_SMARTS)
    has_xanthine = mol.HasSubstructMatch(_XANTHINE_CORE)
    has_triazolo = mol.HasSubstructMatch(_TRIAZOLOTRIAZINE) or mol.HasSubstructMatch(_TRIAZOLOPYRIMIDINE)
    has_quinazoline_urea = mol.HasSubstructMatch(_QUINAZOLINE_UREA)

    if has_carboxamide:
        moa = "High-Efficacy Agonist (Carboxamide-Ribose Engaged)"
        eff_class = "Agonist"
        heuristic_score = 0.95
    elif has_ribose:
        moa = "Agonist / Partial Agonist (Ribose Anchor Present)"
        eff_class = "Agonist"
        heuristic_score = 0.85
    elif has_xanthine or has_triazolo or has_quinazoline_urea:
        moa = "Competitive Antagonist / Inverse Agonist"
        eff_class = "Antagonist"
        heuristic_score = 0.05
    else:
        moa = "Unclassified Non-Canonical Scaffold (Assay Required)"
        eff_class = "Unclassified"
        heuristic_score = 0.50

    if primary_subtype in ("A2A", "A2B"):
        if eff_class == "Agonist":
            pathway = "Gs-coupled -> Adenylyl Cyclase Activation -> cAMP Elevation -> Vasodilation / Anti-inflammatory"
            indication = (
                "Coronary Vasodilation / Immuno-suppression"
                if primary_subtype == "A2A"
                else "Tissue Protection / Anti-fibrotic"
            )
        elif eff_class == "Antagonist":
            pathway = "Gs-blockade -> Basal cAMP Maintenance -> T-cell Activation / Striatal Modulation"
            indication = (
                "Immuno-Oncology Checkpoint Blockade / Parkinson's Disease"
                if primary_subtype == "A2A"
                else "Cancer Immunotherapy / Asthma Relief"
            )
        else:
            pathway = "Gs-modulatory pathway (Subtype A2A/A2B target engagement)"
            indication = "Targeted A2A/A2B Modulation (Functional cAMP validation needed)"
    else:
        if eff_class == "Agonist":
            pathway = "Gi/o-coupled -> Adenylyl Cyclase Inhibition -> cAMP Reduction -> Heart Rate Slowing / Analgesia"
            indication = (
                "Anti-Arrhythmic (AV Node Delay) / Pain Relief"
                if primary_subtype == "A1"
                else "Immuno-modulator / Rheumatoid Arthritis"
            )
        elif eff_class == "Antagonist":
            pathway = "Gi-blockade -> Adenylyl Cyclase Disinhibition -> Renoprotection / Diuresis"
            indication = (
                "Acute Kidney Injury Protection / Diuretic"
                if primary_subtype == "A1"
                else "Inflammatory Disease Relief"
            )
        else:
            pathway = "Gi/o-modulatory pathway (Subtype A1/A3 target engagement)"
            indication = "Targeted A1/A3 Modulation (Functional cAMP validation needed)"

    return {
        "mode_of_action": moa,
        "efficacy_class": eff_class,
        "activation_probability": heuristic_score,
        "signaling_pathway": pathway,
        "therapeutic_indication": indication,
    }
