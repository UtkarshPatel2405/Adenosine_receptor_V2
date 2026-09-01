"""Verified reference presets and authentic GPCRdb structural database catalog."""
from typing import Dict, Any, List

PRESETS: Dict[str, Dict[str, str]] = {
    "Regadenoson (Lexiscan)": {
        "smiles": "CNC(=O)c1cnn(c1)-c1nc(N)c2ncn([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1",
        "name": "Regadenoson (A2A Selective Agonist - FDA Approved)",
    },
    "CGS-21680 (CHEMBL331372)": {
        "smiles": "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O",
        "name": "CGS-21680 (A2A Potent Agonist - Benchmark)",
    },
    "Istradefylline (Nourianz)": {
        "smiles": "CCN1C(=O)C2=C(N=C(N2C)/C=C/c3ccc(OC)c(OC)c3)N(C1=O)CC",
        "name": "Istradefylline (A2A Selective Antagonist)",
    },
    "PSB-603": {
        "smiles": "CCCn1c(=O)c2[nH]c(-c3ccc(cc3)S(=O)(=O)N3CCN(c4ccc(Cl)cc4)CC3)nc2c(=O)n1CCC",
        "name": "PSB-603 (A2B Highly Selective Antagonist)",
    },
    "ZM241385 (4EIY Ligand)": {
        "smiles": "Nc1nc(NCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12",
        "name": "ZM241385 (A2A High-Affinity Antagonist)",
    },
    "CCPA": {
        "smiles": "Clc1nc(NC2CCCC2)c2ncn([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1",
        "name": "CCPA (A1 Potent Selective Agonist)",
    },
    "BAY 60-6583": {
        "smiles": "N#Cc1c(N)nc(-c2ccc(NC(=O)c3cccc(C(F)(F)F)c3)cc2)nc1N",
        "name": "BAY 60-6583 (A2B Selective Agonist)",
    },
    "IB-MECA (CF101)": {
        "smiles": "CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4ccccc4)ncnc32)[C@H](O)[C@@H]1O",
        "name": "IB-MECA (A3 Selective Agonist)",
    },
}

RECEPTOR_STRUCT_DB: Dict[str, Dict[str, Any]] = {
    "A1": {
        "active": {"pdb_id": "6D9H", "method": "Cryo-EM", "resolution": "3.60 Å", "ligand_name": "Adenosine + Gi2", "ccd": "ADE", "title": "Adenosine A1-Gi2 Signaling Complex", "mechanism": "Active canonical G-protein signaling state", "cadd_note": "Orthosteric cavity contracted inward."},
        "inactive": {"pdb_id": "5N2S", "method": "X-ray", "resolution": "3.30 Å", "ligand_name": "PSB36 (Antagonist)", "ccd": "PSB", "title": "Adenosine A1 Inactive Crystal Structure", "mechanism": "Resting ground state xanthine binding mode", "cadd_note": "Expanded extracellular pocket."},
    },
    "A2A": {
        "active": {"pdb_id": "6GDG", "method": "Cryo-EM", "resolution": "4.11 Å", "ligand_name": "Adenosine + mini-Gs", "ccd": "ADE", "title": "Adenosine A2A-miniGs Active Complex", "mechanism": "Full Gs signaling state with outward TM6 swing", "cadd_note": "Canonical Class A GPCR activation hallmark."},
        "inactive": {"pdb_id": "4EIY", "method": "X-ray", "resolution": "1.80 Å", "ligand_name": "ZM241385", "ccd": "ZMA", "title": "Adenosine A2A High-Resolution Inactive State", "mechanism": "Ultra-high resolution antagonist ground state", "cadd_note": "Ordered water network benchmark."},
    },
    "A2B": {
        "active": {"pdb_id": "8HDO", "method": "Cryo-EM", "resolution": "2.87 Å", "ligand_name": "BAY 60-6583 + Gs", "ccd": "BAY", "title": "Adenosine A2B-Gs Complex with Selective Agonist", "mechanism": "Non-nucleoside agonist-stabilized Gs complex", "cadd_note": "Novel subpocket engagement."},
        "inactive": {"pdb_id": "NONE", "method": "AlphaFold DB", "resolution": "N/A (AF-P29275)", "ligand_name": "No Experimental PDB Solved", "ccd": "N/A", "title": "No Experimental Inactive Structure", "mechanism": "No experimental inactive structure in PDB", "cadd_note": "Drug discovery relies on AlphaFold AF-P29275."},
    },
    "A3": {
        "active": {"pdb_id": "8X16", "method": "Cryo-EM", "resolution": "3.29 Å", "ligand_name": "CF101 / IB-MECA + Gi", "ccd": "MECA", "title": "Adenosine A3-Gi Complex with IB-MECA", "mechanism": "Selective immuno-oncology agonist-bound Gi complex", "cadd_note": "Ribose pocket key for A3 selectivity."},
        "inactive": {"pdb_id": "9EHS", "method": "Cryo-EM", "resolution": "3.20 Å", "ligand_name": "LUF7602", "ccd": "LUF", "title": "Adenosine A3 Inactive Structure with Antagonist", "mechanism": "First solved inactive-state A3 ground state", "cadd_note": "Covalent antagonist locking extracellular loop."},
    },
}

GPCRDB_CATALOG_RECORDS: List[Dict[str, str]] = [
    {"Subtype": "Human A1", "State": "Active", "PDB ID": "6D9H", "Method": "Cryo-EM", "Resolution": "3.60 Å", "Ligand / Complex": "Adenosine (Agonist) + Gi2", "Reference": "Endogenous signaling state with TM6 contraction"},
    {"Subtype": "Human A1", "State": "Active", "PDB ID": "7LD3", "Method": "Cryo-EM", "Resolution": "3.20 Å", "Ligand / Complex": "Adenosine (Agonist) + Gi1", "Reference": "High-resolution Gi1 heterotrimer coupling"},
    {"Subtype": "Human A1", "State": "Inactive", "PDB ID": "5N2S", "Method": "X-ray", "Resolution": "3.30 Å", "Ligand / Complex": "PSB36 (Antagonist)", "Reference": "Resting ground-state xanthine conformation"},
    {"Subtype": "Human A1", "State": "Inactive", "PDB ID": "5UEN", "Method": "X-ray", "Resolution": "3.20 Å", "Ligand / Complex": "DU172 (Antagonist)", "Reference": "Orthosteric cavity locked by covalent inhibitor"},
    {"Subtype": "Human A2A", "State": "Active", "PDB ID": "6GDG", "Method": "Cryo-EM", "Resolution": "4.11 Å", "Ligand / Complex": "Adenosine (Agonist) + mini-Gs", "Reference": "Canonical Gs signaling state (~14 Å TM6 swing)"},
    {"Subtype": "Human A2A", "State": "Active", "PDB ID": "2YDO", "Method": "X-ray", "Resolution": "3.00 Å", "Ligand / Complex": "NECA (Agonist)", "Reference": "Thermostabilized purine agonist complex"},
    {"Subtype": "Human A2A", "State": "Inactive", "PDB ID": "4EIY", "Method": "X-ray", "Resolution": "1.80 Å", "Ligand / Complex": "ZM241385 (Antagonist)", "Reference": "Ultra-high resolution benchmark (1.80 Å)"},
    {"Subtype": "Human A2A", "State": "Inactive", "PDB ID": "5IU4", "Method": "X-ray", "Resolution": "1.72 Å", "Ligand / Complex": "ZMA (Antagonist)", "Reference": "Highest resolution Class A GPCR crystal"},
    {"Subtype": "Human A2B", "State": "Active", "PDB ID": "8HDO", "Method": "Cryo-EM", "Resolution": "2.87 Å", "Ligand / Complex": "BAY 60-6583 (Agonist) + Gs", "Reference": "Non-nucleoside agonist-stabilized Gs state"},
    {"Subtype": "Human A2B", "State": "Inactive", "PDB ID": "—", "Method": "AlphaFold DB", "Resolution": "AF-P29275", "Ligand / Complex": "No Experimental PDB", "Reference": "No experimental inactive structure in PDB"},
    {"Subtype": "Human A3", "State": "Active", "PDB ID": "8X16", "Method": "Cryo-EM", "Resolution": "3.29 Å", "Ligand / Complex": "CF101 / IB-MECA (Agonist) + Gi", "Reference": "Clinical immuno-oncology agonist-bound Gi"},
    {"Subtype": "Human A3", "State": "Inactive", "PDB ID": "9EHS", "Method": "Cryo-EM", "Resolution": "3.20 Å", "Ligand / Complex": "LUF7602 (Antagonist)", "Reference": "First solved inactive-state A3 structure"},
]
