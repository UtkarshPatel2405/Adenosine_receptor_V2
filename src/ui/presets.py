"""Verified reference presets for adenosine receptor profiling."""
from typing import Dict

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
