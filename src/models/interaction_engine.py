"""Deterministic 2D SMARTS pharmacophore motif checks for orthosteric pocket hypotheses."""
from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import Lipinski

_EXOCYCLIC_NH2 = Chem.MolFromSmarts("c[NH2]")
_EXOCYCLIC_NH = Chem.MolFromSmarts("c[NH]")
_PURINE_N7 = Chem.MolFromSmarts("n1cnc2ncncc12")
_PURINE_NH = Chem.MolFromSmarts("c1nc2[nH]cnc2n1")
_RIBOSE_MOTIF = Chem.MolFromSmarts("C1OC(C)C(O)C1O")


def analyze_pocket_interactions(smiles: str, primary_subtype: str) -> Dict[str, Any]:
    """Screen 2D SMARTS motifs corresponding to orthosteric adenosine receptor residues."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "asn_anchor_hbond": False,
            "trp_toggle_switch": False,
            "phe_pi_stacking": False,
            "chiral_centers": [],
            "stereocenter_count": 0,
            "chiral_alert": "Invalid structure",
        }

    has_exocyclic_n = mol.HasSubstructMatch(_EXOCYCLIC_NH2) or mol.HasSubstructMatch(_EXOCYCLIC_NH)
    has_hba_nitrogen = any(atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0 for atom in mol.GetAtoms())
    asn_anchor = bool(has_exocyclic_n and has_hba_nitrogen)

    has_purine = mol.HasSubstructMatch(_PURINE_N7) or mol.HasSubstructMatch(_PURINE_NH)
    has_ribose = mol.HasSubstructMatch(_RIBOSE_MOTIF)
    trp_toggle = bool(has_purine and has_ribose)

    phe_stacking = int(Lipinski.NumAromaticRings(mol)) >= 2

    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    chiral_count = len(chiral_centers)
    chiral_alert = (
        "Natural D-enantiomer configuration required for sub-nanomolar potency "
        "(L-enantiomers exhibit >1000x potency loss)."
        if chiral_count >= 3
        else "Achiral or low stereochemical complexity."
    )

    asn_res_name = {
        "A1": "Asn254 (6.55)",
        "A2A": "Asn253 (6.55)",
        "A2B": "Asn254 (6.55)",
        "A3": "Asn250 (6.55)",
    }.get(primary_subtype, "Asn6.55")
    trp_res_name = {
        "A1": "Trp247 (6.48)",
        "A2A": "Trp246 (6.48)",
        "A2B": "Trp247 (6.48)",
        "A3": "Trp243 (6.48)",
    }.get(primary_subtype, "Trp6.48")

    return {
        "asn_anchor_hbond": asn_anchor,
        "asn_residue": asn_res_name,
        "trp_toggle_switch": trp_toggle,
        "trp_residue": trp_res_name,
        "phe_pi_stacking": phe_stacking,
        "chiral_centers": chiral_centers,
        "stereocenter_count": chiral_count,
        "chiral_alert": chiral_alert,
    }
