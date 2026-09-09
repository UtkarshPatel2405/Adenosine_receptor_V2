"""Pillar 1: Functional Efficacy & Mode of Action (MoA) Engine."""
from typing import Dict, Any
from rdkit import Chem

# Structural SMART patterns defining GPCR activation hallmarks
_RIBOSE_SMARTS = Chem.MolFromSmarts("C1OC(CO)C(O)C1O")
_CARBOXAMIDE_RIBOSE = Chem.MolFromSmarts("C1OC(C(=O)N)C(O)C1O")
_XANTHINE_CORE = Chem.MolFromSmarts("c1nc2c(=O)[nH]c(=O)n(C)c2n1")
_TRIAZOLOPYRIMIDINE = Chem.MolFromSmarts("c1nnc2nc(NCc3ccccc3)nc2n1")
_PURINE_CORE = Chem.MolFromSmarts("c1nc2ncnc2n1")


def predict_functional_efficacy(smiles: str, primary_subtype: str, max_pchembl: float, in_domain: bool = True) -> Dict[str, Any]:
    """Classify functional efficacy mode of action and secondary messenger cascade."""
    if not in_domain or max_pchembl < 5.0:
        return {
            "mode_of_action": "Inactive / Non-binder",
            "efficacy_class": "Non-functional",
            "activation_probability": 0.0,
            "signaling_pathway": "No G-protein coupling",
            "therapeutic_indication": "Inactive",
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mode_of_action": "Unknown", "efficacy_class": "Unknown", "activation_probability": 0.0, "signaling_pathway": "None", "therapeutic_indication": "N/A"}

    has_ribose = mol.HasSubstructMatch(_RIBOSE_SMARTS) or mol.HasSubstructMatch(_CARBOXAMIDE_RIBOSE)
    has_carboxamide = mol.HasSubstructMatch(_CARBOXAMIDE_RIBOSE)
    has_xanthine = mol.HasSubstructMatch(_XANTHINE_CORE)
    has_triazolo = mol.HasSubstructMatch(_TRIAZOLOPYRIMIDINE)

    # Structural MoA classification based on conserved GPCR activation hallmarks
    if has_carboxamide:
        moa = "High-Efficacy Agonist (Carboxamide-Ribose Engaged)"
        eff_class = "Agonist"
        act_prob = 0.95
    elif has_ribose:
        moa = "Agonist / Partial Agonist (Ribose Anchor Present)"
        eff_class = "Agonist"
        act_prob = 0.85
    elif has_xanthine or has_triazolo:
        moa = "Competitive Antagonist / Inverse Agonist"
        eff_class = "Antagonist"
        act_prob = 0.05
    else:
        # Non-ribose heterocyclic chemotype
        moa = "Unclassified Heterocyclic Chemotype (Assay Required)"
        eff_class = "Unclassified"
        act_prob = 0.50

    # G-protein signaling cascades by subtype
    if primary_subtype in ("A2A", "A2B"):
        if eff_class == "Agonist":
            pathway = "Gs-coupled -> Adenylyl Cyclase Activation -> cAMP Elevation -> Vasodilation / Anti-inflammatory"
            indication = "Coronary Vasodilation / Immuno-suppression" if primary_subtype == "A2A" else "Tissue Protection / Anti-fibrotic"
        elif eff_class == "Antagonist":
            pathway = "Gs-blockade -> Basal cAMP Maintenance -> T-cell Activation / Striatal Modulation"
            indication = "Immuno-Oncology Checkpoint Blockade / Parkinson's Disease" if primary_subtype == "A2A" else "Cancer Immunotherapy / Asthma Relief"
        else:
            pathway = "Gs-modulatory pathway (Subtype A2A/A2B target engagement)"
            indication = "Targeted A2A/A2B Modulation (Functional cAMP validation needed)"
    else:  # A1, A3
        if eff_class == "Agonist":
            pathway = "Gi/o-coupled -> Adenylyl Cyclase Inhibition -> cAMP Reduction -> Heart Rate Slowing / Analgesia"
            indication = "Anti-Arrhythmic (AV Node Delay) / Pain Relief" if primary_subtype == "A1" else "Immuno-modulator / Rheumatoid Arthritis"
        elif eff_class == "Antagonist":
            pathway = "Gi-blockade -> Adenylyl Cyclase Disinhibition -> Renoprotection / Diuresis"
            indication = "Acute Kidney Injury Protection / Diuretic" if primary_subtype == "A1" else "Inflammatory Disease Relief"
        else:
            pathway = "Gi/o-modulatory pathway (Subtype A1/A3 target engagement)"
            indication = "Targeted A1/A3 Modulation (Functional cAMP validation needed)"

    return {
        "mode_of_action": moa,
        "efficacy_class": eff_class,
        "activation_probability": act_prob,
        "signaling_pathway": pathway,
        "therapeutic_indication": indication,
    }

