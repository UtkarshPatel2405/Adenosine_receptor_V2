import json
from pathlib import Path

def patch_registry():
    """Patch adenosine PDB ligand registry with curated structures.
    
    Adds manually verified A3 structures (professor-curated) and
    missing A2A entries not captured by the RCSB API search.
    """
    json_path = Path("data/processed/adenosine_pdb_ligands.json")
    if json_path.exists():
        with open(json_path) as f:
            registry = json.load(f)
    else:
        registry = {}
    
    # ── A3: Professor-curated structures ──────────────────────────────
    # Only 2 existed before (7LD3, 8J78). Professor provided 5 additional.
    registry["A3"] = [
        {
            "pdb_id": "8X16",
            "ligands": [
                {
                    "ccd": "Q8L",
                    "name": "Piclidenoson (CF101, IB-MECA)",
                    "smiles": "CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4cccc(I)c4)ncnc32)[C@H](O)[C@@H]1O",
                    "formula": "C18 H19 I N6 O4",
                    "mw": 510.28
                }
            ]
        },
        {
            "pdb_id": "8X17",
            "ligands": [
                {
                    "ccd": "XS0",
                    "name": "Namodenoson (CF102, 2-Cl-IB-MECA)",
                    "smiles": "CNC(=O)[C@@H]1[C@H]([C@H]([C@@H](O1)n2cnc3c2nc(nc3NCc4cccc(I)c4)Cl)O)O",
                    "formula": "C18 H18 Cl I N6 O4",
                    "mw": 544.72
                }
            ]
        },
        {
            "pdb_id": "8YH2",
            "ligands": [
                {
                    "ccd": "ADN",
                    "name": "Adenosine",
                    "smiles": "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)CO)O)O)N",
                    "formula": "C10 H13 N5 O4",
                    "mw": 267.241
                }
            ]
        },
        {
            "pdb_id": "9EBH",
            "ligands": [
                {
                    "ccd": "ADN",
                    "name": "Adenosine",
                    "smiles": "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)CO)O)O)N",
                    "formula": "C10 H13 N5 O4",
                    "mw": 267.241
                }
            ]
        },
        {
            "pdb_id": "9EBI",
            "ligands": [
                {
                    "ccd": "Q8L",
                    "name": "Piclidenoson (CF101, IB-MECA)",
                    "smiles": "CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4cccc(I)c4)ncnc32)[C@H](O)[C@@H]1O",
                    "formula": "C18 H19 I N6 O4",
                    "mw": 510.28
                }
            ]
        },
    ]
    
    # ── A2A: Append 8RLN if missing ───────────────────────────────────
    if "A2A" not in registry:
        registry["A2A"] = []
        
    if not any(x.get("pdb_id") == "8RLN" for x in registry["A2A"]):
        registry["A2A"].append({
            "pdb_id": "8RLN",
            "ligands": [
                {
                    "ccd": "A1H1S",
                    "name": "LUF5834 (partial agonist)",
                    "smiles": "NC1=NC(SCC3=NC=CN3)=C(C#N)C(C2=CC=C(O)C=C2)=C1C#N",
                    "formula": "C17 H12 N6 O S",
                    "mw": 348.38
                }
            ]
        })
    
    with open(json_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    # Report
    for sub in ("A1", "A2A", "A2B", "A3"):
        n = len(registry.get(sub, []))
        print(f"  {sub}: {n} PDB entries")
    print("Successfully patched registry!")

if __name__ == "__main__":
    patch_registry()
