"""OECD Principle 3: Quantitative Applicability Domain (AD) Assessment."""
import logging
import pickle
from functools import lru_cache
from typing import Any, Dict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.config import MORGAN_BITS, MORGAN_RADIUS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

_MORGAN_GEN = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

_FALLBACK_REFERENCE_SMILES = {
    "Adenosine": "Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O",
    "ZM241385": "Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12",
    "CGS-21680": "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O",
    "Istradefylline": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
    "PSB-603": "CCCn1c(=O)[nH]c2nc(-c3ccc(S(=O)(=O)N4CCN(c5ccc(Cl)cc5)CC4)cc3)[nH]c2c1=O",
    "BAY 60-6583": "N#Cc1c(N)nc(SCC(N)=O)c(C#N)c1-c1ccc(OCC2CC2)cc1",
    "IB-MECA": "CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4cccc(I)c4)ncnc32)[C@H](O)[C@@H]1O",
    "CCPA": "OC[C@H]1O[C@@H](n2cnc3c(NC4CCCC4)nc(Cl)nc32)[C@H](O)[C@@H]1O",
}


def _build_fallback_fps():
    fps = []
    for smi in _FALLBACK_REFERENCE_SMILES.values():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(_MORGAN_GEN.GetFingerprint(mol))
    return fps


_FALLBACK_REF_FPS = _build_fallback_fps()


@lru_cache(maxsize=2)
def _load_cached_train_fps(path_str: str, mtime_ns: int):
    with open(path_str, "rb") as f:
        fps = pickle.load(f)
    return [fp for fp in fps if fp is not None]


def _get_reference_fps() -> tuple[list, str]:
    fps_path = PROCESSED_DATA_DIR / "train_fps.pkl"
    if fps_path.exists():
        try:
            fps = _load_cached_train_fps(str(fps_path), fps_path.stat().st_mtime_ns)
            if fps:
                return fps, f"train_fps.pkl ({len(fps)} training compounds)"
        except Exception as exc:
            logger.warning("Failed to load %s, using fallback probes: %s", fps_path, exc)
    return _FALLBACK_REF_FPS, f"fallback_reference_probes ({len(_FALLBACK_REF_FPS)} compounds)"


def check_applicability_domain(smiles: str) -> Dict[str, Any]:
    """Evaluate OECD Principle 3 Applicability Domain via Morgan Tanimoto and property gates."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {
            "in_domain": False,
            "status": "Invalid SMILES",
            "domain_status": "Invalid SMILES",
            "tanimoto_max": 0.0,
            "reference_source": "none",
            "property_violations": ["Invalid molecular structure"],
            "violations": ["Invalid molecular structure"],
        }

    fp = _MORGAN_GEN.GetFingerprint(mol)
    ref_fps, ref_source = _get_reference_fps()
    sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps) if ref_fps else [0.0]
    max_tanimoto = float(np.max(sims)) if sims else 0.0

    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))

    violations = []
    if mw < 120 or mw > 850:
        violations.append(f"MW {mw:.1f} outside [120, 850]")
    if logp < -3.0 or logp > 7.0:
        violations.append(f"LogP {logp:.2f} outside [-3.0, 7.0]")
    if tpsa > 250:
        violations.append(f"TPSA {tpsa:.1f} > 250 A^2")
    if rotb > 18:
        violations.append(f"Rotatable bonds {rotb} > 18")

    property_gates_ok = len(violations) == 0

    if max_tanimoto >= 0.35 and property_gates_ok:
        status = "Inside AD (High Confidence)"
        in_ad = True
        in_ad_label = "Yes"
    elif 0.25 <= max_tanimoto < 0.35 and len(violations) <= 1:
        status = "Borderline AD (Moderate Confidence / Scaffold Hop)"
        in_ad = True
        in_ad_label = "Borderline (Moderate Confidence / Scaffold Hop)"
    elif max_tanimoto >= 0.35 and len(violations) == 1:
        status = "Borderline AD (Property Gate Advisory)"
        in_ad = True
        in_ad_label = "Borderline (Property Gate Advisory)"
    else:
        status = "Outside AD (Extrapolation Warning)"
        in_ad = False
        in_ad_label = "No (Extrapolation Warning)"

    return {
        "in_domain": in_ad,
        "in_domain_label": in_ad_label,
        "status": status,
        "domain_status": status,
        "tanimoto_max": round(max_tanimoto, 3),
        "reference_source": ref_source,
        "property_violations": violations,
        "violations": violations,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
    }
