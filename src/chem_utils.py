from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, QED, rdDistGeom, rdForceFieldHelpers, rdPartialCharges
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.config import MORGAN_BITS, MORGAN_RADIUS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    _FILTER_CATALOG_AVAILABLE = True
except ImportError:
    _FILTER_CATALOG_AVAILABLE = False

_MORGAN = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()


@lru_cache(maxsize=10000)
def canonicalize(smiles: str) -> Optional[str]:
    """Parse SMILES, strip salts, neutralize charges, and return canonical isomeric SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as exc:
        logger.warning("SMILES standardization failed for %r: %s", smiles, exc)
        return None


def mol_from_smiles(smiles: str):
    canon = canonicalize(smiles)
    if canon is None:
        return None
    return Chem.MolFromSmiles(canon)


def draw_2d(
    smiles: str,
    size: tuple[int, int] = (400, 300),
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    if width is not None and height is not None:
        size = (width, height)
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import Draw, rdDepictor
        rdDepictor.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=size)
    except Exception as exc:
        logger.warning("2D image rendering failed: %s", exc)
        return None


def draw_2d_svg(
    smiles: str,
    size: tuple[int, int] = (400, 300),
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    if width is not None and height is not None:
        size = (width, height)
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.warning("SVG drawing failed, falling back to raster image: %s", exc)
    try:
        from rdkit.Chem import Draw
        return Draw.MolToImage(mol, size=size)
    except Exception as exc:
        logger.warning("Fallback 2D raster rendering failed: %s", exc)
        return None


generate_2d_svg = draw_2d_svg


def _extract_gasteiger_bounds(mol: Chem.Mol) -> tuple[float, float]:
    charges = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_GasteigerCharge"):
            try:
                c = float(atom.GetProp("_GasteigerCharge"))
                if not np.isnan(c) and not np.isinf(c):
                    charges.append(c)
            except ValueError:
                continue
    return (min(charges), max(charges)) if charges else (0.0, 0.0)


def _embed_3d_mol(mol: Chem.Mol) -> Chem.Mol:
    mol_3d = Chem.AddHs(mol)
    embed_status = rdDistGeom.EmbedMolecule(mol_3d, rdDistGeom.ETKDGv3())
    if embed_status != 0:
        embed_status = rdDistGeom.EmbedMolecule(mol_3d)
    if embed_status == 0:
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol_3d)
    return mol_3d


def generate_3d_conformer(smiles: str) -> tuple[Optional[str], float, float]:
    """Generate an ETKDGv3/MMFF94 3D conformer and Gasteiger partial charge bounds."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, 0.0, 0.0
    try:
        mol_3d = _embed_3d_mol(mol)
        rdPartialCharges.ComputeGasteigerCharges(mol_3d)
        min_charge, max_charge = _extract_gasteiger_bounds(mol_3d)
        return Chem.MolToMolBlock(mol_3d), min_charge, max_charge
    except Exception as exc:
        logger.warning("3D conformer generation failed, falling back to 2D charges: %s", exc)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            min_charge, max_charge = _extract_gasteiger_bounds(mol)
            return None, min_charge, max_charge
        except Exception as charge_exc:
            logger.warning("Gasteiger charge calculation failed: %s", charge_exc)
            return None, 0.0, 0.0


@lru_cache(maxsize=1)
def _build_pains_catalog():
    if not _FILTER_CATALOG_AVAILABLE:
        return None
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def check_pains(smiles: str) -> list[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    catalog = _build_pains_catalog()
    if catalog is None:
        return []
    return [entry.GetDescription() for entry in catalog.GetMatches(mol)]


def qed_profile(smiles: str) -> Optional[dict]:
    """Compute QED, Lipinski descriptors, and Veber oral bioavailability rules."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    rotbonds = int(Lipinski.NumRotatableBonds(mol))
    tpsa = round(float(Descriptors.TPSA(mol)), 2)
    veber_tpsa_ok = bool(tpsa <= 140.0)
    veber_rotbonds_ok = bool(rotbonds <= 10)
    veber_violations = int(not veber_tpsa_ok) + int(not veber_rotbonds_ok)
    return {
        "QED": round(QED.qed(mol), 4),
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotB": rotbonds,
        "AromaticRings": int(Lipinski.NumAromaticRings(mol)),
        "TPSA": tpsa,
        "veber_tpsa_ok": veber_tpsa_ok,
        "veber_rotbonds_ok": veber_rotbonds_ok,
        "veber_violations": veber_violations,
        "veber_pass": bool(veber_violations == 0),
    }


@lru_cache(maxsize=2)
def _load_pickle_cached(path_str: str, mtime_ns: int):
    with open(path_str, "rb") as f:
        return pickle.load(f)


def _load_train_fps():
    path = PROCESSED_DATA_DIR / "train_fps.pkl"
    return _load_pickle_cached(str(path), path.stat().st_mtime_ns)


def _load_train_smiles() -> list[str]:
    path = PROCESSED_DATA_DIR / "train_smiles.pkl"
    return _load_pickle_cached(str(path), path.stat().st_mtime_ns)


def nearest_tanimoto(smiles: str) -> Optional[float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        train_fps = _load_train_fps()
    except FileNotFoundError:
        return None
    qfp = _MORGAN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
    return float(np.max(sims)) if sims else None


def topk_tanimoto(smiles: str, k: int = 10) -> tuple[Optional[str], list[tuple[str, float]]]:
    canon = canonicalize(smiles)
    if canon is None:
        return None, []
    mol = Chem.MolFromSmiles(canon)
    qfp = _MORGAN.GetFingerprint(mol)
    try:
        train_fps = _load_train_fps()
        train_smiles = _load_train_smiles()
    except FileNotFoundError:
        return canon, []
    sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
    idx = np.argsort(sims)[::-1]
    seen = set()
    top = []
    for i in idx:
        smi = train_smiles[i]
        if smi not in seen:
            seen.add(smi)
            top.append((smi, float(sims[i])))
            if len(top) >= k:
                break
    return canon, top


def generate_pdb_block(smiles: str) -> Optional[str]:
    """Generate a 3D conformer and return a PDB block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToPDBBlock(_embed_3d_mol(mol))
    except Exception as exc:
        logger.warning("PDB block generation failed: %s", exc)
        return None


def generate_sdf_block(smiles: str) -> Optional[str]:
    """Generate a 3D conformer and return an SDF/Mol block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToMolBlock(_embed_3d_mol(mol))
    except Exception as exc:
        logger.warning("SDF block generation failed: %s", exc)
        return None


def generate_2d_mol_block(smiles: str) -> Optional[str]:
    """Generate 2D coordinates and return a Mol block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)
    except Exception as exc:
        logger.warning("2D Mol block generation failed: %s", exc)
        return None
