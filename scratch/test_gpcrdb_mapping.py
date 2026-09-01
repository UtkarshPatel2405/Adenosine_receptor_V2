import sys
from src.pdb_utils import find_gpcrdb_structure_matches, real_structure_refs_with_analogs
from src.api_routes.analysis import receptor_neighbors

test_ligands = {
    "CGS-21680": "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O",
    "ZM241385": "Nc1nc(NCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12",
    "BAY 60-6583": "N#Cc1c(N)nc(-c2ccc(NC(=O)c3cccc(C(F)(F)F)c3)cc2)nc1N",
    "IB-MECA (CF101)": "CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4ccccc4)ncnc32)[C@H](O)[C@@H]1O",
    "CCPA": "Clc1nc(NC2CCCC2)c2ncn([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1",
}

print("=== GPCRdb Structure Mapping Test ===")
for name, smi in test_ligands.items():
    matches = find_gpcrdb_structure_matches(smi)
    print(f"\nLigand: {name}")
    if matches:
        for m in matches[:3]:
            print(f"  -> Match PDB: {m['pdb_id']} (Subtype: {m['subtype']}, State: {m['state']}, Resolution: {m['resolution']}, Ligand: {m['ligand_name']}, Match: {m['match_type']})")
    else:
        print("  -> No matching GPCRdb structure template.")

print("\n=== Receptor Training Neighbors with GPCRdb PDB Test ===")
nbrs = receptor_neighbors(test_ligands["CGS-21680"], "A2A", top_k=3)
if nbrs:
    for n in nbrs:
        print(f"Neighbor SMILES: {n['smiles'][:30]}... | Tanimoto: {n['tanimoto']} | Activity: {n.get('activity')} | PDBs: {[r['id'] for r in n['real_structures']]}")
