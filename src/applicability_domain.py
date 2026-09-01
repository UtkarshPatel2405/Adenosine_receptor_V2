"""OECD Principle 3: Quantitative Applicability Domain (AD) Assessment."""
from typing import Dict, Any, List, Optional
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski

from src.config import PROCESSED_DATA_DIR

# Reference canonical adenosine pharmacophore chemotypes for fast bulk similarity
_REFERENCE_SMILES = [
    "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)CO)O)O)N",  # Adenosine
    "Nc1nc(NCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12",  # ZM241385
    "CCNC(=O)C1OC(n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)C(O)C1O",  # CGS-21680
    "CCN1C(=O)C2=C(N=C(N2C)/C=C/c3ccc(OC)c(OC)c3)N(C1=O)CC",  # Istradefylline
    "CCCn1c(=O)c2[nH]c(-c3ccc(cc3)S(=O)(=O)N3CCN(c4ccc(Cl)cc4)CC3)nc2c(=O)n1CCC",  # PSB-603
    "Clc1nc(NC2CCCC2)c2ncn(C3OC(CO)C(O)C3O)c2n1",  # CCPA
    "N#Cc1c(N)nc(-c2ccc(NC(=O)c3cccc(C(F)(F)F)c3)cc2)nc1N",  # BAY 60-6583
    "CNC(=O)C1OC(n2cnc3c(NCc4ccccc4)ncnc32)C(O)C1O",  # IB-MECA
]

_REF_FPS = []
for _s in _REFERENCE_SMILES:
    _m = Chem.MolFromSmiles(_s)
    if _m:
        _REF_FPS.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(_m, 2, nBits=2048))


def check_applicability_domain(smiles: str) -> Dict[str, Any]:
    """Evaluate OECD Principle 3 Applicability Domain via Morgan fingerprint Tanimoto & physicochemical bounds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"in_domain": False, "domain_status": "Invalid SMILES", "tanimoto_max": 0.0, "violations": ["Invalid molecular structure"]}

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, _REF_FPS) if _REF_FPS else [0.0]
    max_tanimoto = float(np.max(sims)) if sims else 0.0

    # Physicochemical boundary checks
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))

    violations = []
    if mw < 120 or mw > 850: violations.append(f"MW {mw:.1f} outside [120, 850]")
    if logp < -3.0 or logp > 7.0: violations.append(f"LogP {logp:.2f} outside [-3.0, 7.0]")
    if tpsa > 250: violations.append(f"TPSA {tpsa:.1f} > 250 Å²")
    if rotb > 18: violations.append(f"Rotatable bonds {rotb} > 18")

    # Classification logic based on topological similarity and property space
    if max_tanimoto >= 0.35 and len(violations) == 0:
        status = "Inside AD (High Confidence)"
        in_ad = True
    elif max_tanimoto >= 0.25 and len(violations) <= 1:
        status = "Borderline AD (Moderate Confidence / Scaffold Hop)"
        in_ad = True
    else:
        status = "Outside AD (Extrapolation Warning)"
        in_ad = False

    return {
        "in_domain": in_ad,
        "domain_status": status,
        "tanimoto_max": round(max_tanimoto, 3),
        "violations": violations,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
    }
