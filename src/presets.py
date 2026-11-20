"""Reference ligand presets and GPCRdb structural catalog."""
from typing import Any, Dict, List

PRESETS: Dict[str, Dict[str, str]] = {
    "Regadenoson (Lexiscan)": {
        "smiles": "CNC(=O)c1cnn(-c2nc(N)c3ncn([C@@H]4O[C@H](CO)[C@@H](O)[C@H]4O)c3n2)c1",
        "name": "Regadenoson (A2A Selective Agonist - FDA Approved) [TRAIN]",
        "split": "training",
    },
    "CGS-21680": {
        "smiles": "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O",
        "name": "CGS-21680 (A2A Selective Agonist) [TRAIN]",
        "split": "training",
    },
    "Istradefylline": {
        "smiles": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
        "name": "Istradefylline (A2A Selective Antagonist - FDA Approved) [TRAIN]",
        "split": "training",
    },
    "PSB-603": {
        "smiles": "CCCn1c(=O)[nH]c2nc(-c3ccc(S(=O)(=O)N4CCN(c5ccc(Cl)cc5)CC4)cc3)[nH]c2c1=O",
        "name": "PSB-603 (A2B Selective Antagonist) [TRAIN]",
        "split": "training",
    },
    "ZM241385": {
        "smiles": "Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12",
        "name": "ZM241385 (A2A Reference Antagonist) [TRAIN]",
        "split": "training",
    },
    "CCPA": {
        "smiles": "OC[C@H]1O[C@@H](n2cnc3c(NC4CCCC4)nc(Cl)nc32)[C@H](O)[C@@H]1O",
        "name": "CCPA (A1 Selective Agonist) [HOLDOUT TEST]",
        "split": "test",
    },
    "BAY 60-6583": {
        "smiles": "N#Cc1c(N)nc(SCC(N)=O)c(C#N)c1-c1ccc(OCC2CC2)cc1",
        "name": "BAY 60-6583 (A2B Selective Agonist) [HOLDOUT TEST]",
        "split": "test",
    },
    "IB-MECA (CF101)": {
        "smiles": "CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4cccc(I)c4)ncnc32)[C@H](O)[C@@H]1O",
        "name": "IB-MECA / CF101 (A3 Selective Agonist) [TRAIN]",
        "split": "training",
    },
    "Caffeine": {
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "name": "Caffeine (Classic Non-Selective Xanthine Antagonist) [TRAIN]",
        "split": "training",
    },
    "Theophylline": {
        "smiles": "Cn1c2nc[nH]c2c(=O)n(C)c1=O",
        "name": "Theophylline (Classic Xanthine Antagonist) [TRAIN]",
        "split": "training",
    },
    "NECA": {
        "smiles": "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
        "name": "NECA (Pan-Adenosine Nucleoside Agonist) [HOLDOUT TEST]",
        "split": "test",
    },
    "DPCPX": {
        "smiles": "CCCn1c(=O)c2[nH]c(C3CCCC3)nc2n(CCC)c1=O",
        "name": "DPCPX (Classic A1 Selective Xanthine Antagonist) [TRAIN]",
        "split": "training",
    },
}

RECEPTOR_STRUCT_DB: Dict[str, Dict[str, Any]] = {
    "A1": {
        "active": {
            "pdb_id": "6D9H",
            "method": "Cryo-EM",
            "resolution": "3.60 A",
            "ligand_name": "Adenosine + Gi2",
            "ccd": "ADE",
            "title": "Adenosine A1-Gi2 Signaling Complex",
            "mechanism": "Active canonical G-protein signaling state",
            "cadd_note": "Orthosteric cavity contracted inward.",
        },
        "inactive": {
            "pdb_id": "5N2S",
            "method": "X-ray",
            "resolution": "3.30 A",
            "ligand_name": "PSB36 (Antagonist)",
            "ccd": "PSB",
            "title": "Adenosine A1 Inactive Crystal Structure",
            "mechanism": "Resting ground state xanthine binding mode",
            "cadd_note": "Expanded extracellular pocket.",
        },
    },
    "A2A": {
        "active": {
            "pdb_id": "6GDG",
            "method": "Cryo-EM",
            "resolution": "4.11 A",
            "ligand_name": "Adenosine + mini-Gs",
            "ccd": "ADE",
            "title": "Adenosine A2A-miniGs Active Complex",
            "mechanism": "Full Gs signaling state with outward TM6 swing",
            "cadd_note": "Canonical Class A GPCR activation hallmark.",
        },
        "inactive": {
            "pdb_id": "4EIY",
            "method": "X-ray",
            "resolution": "1.80 A",
            "ligand_name": "ZM241385",
            "ccd": "ZMA",
            "title": "Adenosine A2A High-Resolution Inactive State",
            "mechanism": "High-resolution antagonist ground state",
            "cadd_note": "Ordered water network benchmark.",
        },
    },
    "A2B": {
        "active": {
            "pdb_id": "8HDO",
            "method": "Cryo-EM",
            "resolution": "2.87 A",
            "ligand_name": "BAY 60-6583 + Gs",
            "ccd": "BAY",
            "title": "Adenosine A2B-Gs Complex with Selective Agonist",
            "mechanism": "Non-nucleoside agonist-stabilized Gs complex",
            "cadd_note": "Novel subpocket engagement.",
        },
        "inactive": {
            "pdb_id": "NONE",
            "method": "AlphaFold DB",
            "resolution": "N/A (AF-P29275)",
            "ligand_name": "No Experimental PDB Solved",
            "ccd": "N/A",
            "title": "No Experimental Inactive Structure",
            "mechanism": "No experimental inactive structure in PDB",
            "cadd_note": "Structural modeling relies on AlphaFold AF-P29275.",
        },
    },
    "A3": {
        "active": {
            "pdb_id": "8X16",
            "method": "Cryo-EM",
            "resolution": "3.29 A",
            "ligand_name": "CF101 / IB-MECA + Gi",
            "ccd": "MECA",
            "title": "Adenosine A3-Gi Complex with IB-MECA",
            "mechanism": "Selective agonist-bound Gi complex",
            "cadd_note": "Ribose pocket key for A3 selectivity.",
        },
        "inactive": {
            "pdb_id": "9EHS",
            "method": "Cryo-EM",
            "resolution": "3.20 A",
            "ligand_name": "LUF7602",
            "ccd": "LUF",
            "title": "Adenosine A3 Inactive Structure with Antagonist",
            "mechanism": "Inactive-state A3 ground state",
            "cadd_note": "Covalent antagonist locking extracellular loop.",
        },
    },
}

GPCRDB_CATALOG_RECORDS: List[Dict[str, str]] = [
    {"Subtype": "Human A1", "State": "Active", "PDB ID": "6D9H", "Method": "Cryo-EM", "Resolution": "3.60 A", "Ligand / Complex": "Adenosine (Agonist) + Gi2", "Reference": "Endogenous signaling state with TM6 contraction"},
    {"Subtype": "Human A1", "State": "Active", "PDB ID": "7LD3", "Method": "Cryo-EM", "Resolution": "3.20 A", "Ligand / Complex": "Adenosine (Agonist) + Gi1", "Reference": "High-resolution Gi1 heterotrimer coupling"},
    {"Subtype": "Human A1", "State": "Inactive", "PDB ID": "5N2S", "Method": "X-ray", "Resolution": "3.30 A", "Ligand / Complex": "PSB36 (Antagonist)", "Reference": "Resting ground-state xanthine conformation"},
    {"Subtype": "Human A1", "State": "Inactive", "PDB ID": "5UEN", "Method": "X-ray", "Resolution": "3.20 A", "Ligand / Complex": "DU172 (Antagonist)", "Reference": "Orthosteric cavity locked by covalent inhibitor"},
    {"Subtype": "Human A2A", "State": "Active", "PDB ID": "6GDG", "Method": "Cryo-EM", "Resolution": "4.11 A", "Ligand / Complex": "Adenosine (Agonist) + mini-Gs", "Reference": "Canonical Gs signaling state (~14 A TM6 swing)"},
    {"Subtype": "Human A2A", "State": "Active", "PDB ID": "2YDO", "Method": "X-ray", "Resolution": "3.00 A", "Ligand / Complex": "NECA (Agonist)", "Reference": "Thermostabilized purine agonist complex"},
    {"Subtype": "Human A2A", "State": "Inactive", "PDB ID": "4EIY", "Method": "X-ray", "Resolution": "1.80 A", "Ligand / Complex": "ZM241385 (Antagonist)", "Reference": "High-resolution crystal benchmark (1.80 A)"},
    {"Subtype": "Human A2A", "State": "Inactive", "PDB ID": "5IU4", "Method": "X-ray", "Resolution": "1.72 A", "Ligand / Complex": "ZMA (Antagonist)", "Reference": "High-resolution Class A GPCR crystal"},
    {"Subtype": "Human A2B", "State": "Active", "PDB ID": "8HDO", "Method": "Cryo-EM", "Resolution": "2.87 A", "Ligand / Complex": "BAY 60-6583 (Agonist) + Gs", "Reference": "Non-nucleoside agonist-stabilized Gs state"},
    {"Subtype": "Human A2B", "State": "Inactive", "PDB ID": "N/A", "Method": "AlphaFold DB", "Resolution": "AF-P29275", "Ligand / Complex": "No Experimental PDB", "Reference": "No experimental inactive structure in PDB"},
    {"Subtype": "Human A3", "State": "Active", "PDB ID": "8X16", "Method": "Cryo-EM", "Resolution": "3.29 A", "Ligand / Complex": "CF101 / IB-MECA (Agonist) + Gi", "Reference": "Agonist-bound Gi signaling complex"},
    {"Subtype": "Human A3", "State": "Inactive", "PDB ID": "9EHS", "Method": "Cryo-EM", "Resolution": "3.20 A", "Ligand / Complex": "LUF7602 (Antagonist)", "Reference": "Inactive-state A3 crystal/cryo-EM structure"},
]
