"""Pillar 2: 3D Pocket Interaction Fingerprints & Stereocenter Analysis."""
from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

_EXOCYCLIC_AMINE_SMARTS = Chem.MolFromSmarts("c1nc(N)c2ncn(C)c2n1")
_PURINE_N7_SMARTS = Chem.MolFromSmarts("n1cnc2ncncc12")
_AROMATIC_RING_SMARTS = Chem.MolFromSmarts("a1aaaaa1")


def analyze_pocket_interactions(smiles: str, primary_subtype: str) -> Dict[str, Any]:
    """Analyze 3D interaction anchors, toggle switch engagement, and chiral centers."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"asn_anchor_hbond": False, "trp_toggle_switch": False, "phe_pi_stacking": False, "chiral_centers": [], "stereocenter_count": 0, "chiral_alert": "Invalid structure"}

    # 1. Asn6.55 Dual H-Bond Anchor (Asn253 in A2A, Asn254 in A1, Asn250 in A3)
    has_exocyclic_nh2 = mol.HasSubstructMatch(Chem.MolFromSmarts("c[NH2]")) or mol.HasSubstructMatch(Chem.MolFromSmarts("c[NH]"))
    has_hba_nitrogen = any(atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0 for atom in mol.GetAtoms())
    asn_anchor = bool(has_exocyclic_nh2 and has_hba_nitrogen)

    # 2. Trp6.48 Toggle Switch (Orthosteric steric transmission to TM6 outward swing)
    has_purine = mol.HasSubstructMatch(_PURINE_N7_SMARTS) or mol.HasSubstructMatch(Chem.MolFromSmarts("c1nc2[nH]cnc2n1"))
    has_ribose = mol.HasSubstructMatch(Chem.MolFromSmarts("C1OC(C)C(O)C1O"))
    trp_toggle = bool(has_purine and has_ribose)

    # 3. Phe168 (ECL2) Pi-Pi Stacking
    num_arom_rings = int(Lipinski.NumAromaticRings(mol))
    phe_stacking = num_arom_rings >= 2

    # 4. Chiral Center Inventory
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    chiral_count = len(chiral_centers)
    
    # Stereochemical potency cliff detection (D-ribose vs L-ribose)
    chiral_alert = "Natural D-enantiomer configuration required for sub-nanomolar potency (L-enantiomers exhibit >1000x potency loss)." if chiral_count >= 3 else "Achiral or low stereochemical complexity."

    asn_res_name = {"A1": "Asn254 (6.55)", "A2A": "Asn253 (6.55)", "A2B": "Asn254 (6.55)", "A3": "Asn250 (6.55)"}.get(primary_subtype, "Asn6.55")
    trp_res_name = {"A1": "Trp247 (6.48)", "A2A": "Trp246 (6.48)", "A2B": "Trp247 (6.48)", "A3": "Trp243 (6.48)"}.get(primary_subtype, "Trp6.48")

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
