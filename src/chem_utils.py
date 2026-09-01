from __future__ import annotations
import json
import pickle
import os
from functools import lru_cache
from typing import Optional
from pathlib import Path 

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, QED, rdDistGeom, rdForceFieldHelpers, rdPartialCharges
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    _FILTER_CATALOG_AVAILABLE = True
except ImportError:
    _FILTER_CATALOG_AVAILABLE = False

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from src.config import PROCESSED_DATA_DIR

@lru_cache(maxsize=10000)
def canonicalize(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def mol_from_smiles(smiles: str):
    canon = canonicalize(smiles)
    if canon is None:
        return None
    return Chem.MolFromSmiles(canon)

def draw_2d(smiles: str, size: tuple[int, int] = (400, 300), width: Optional[int] = None, height: Optional[int] = None):
    if width is not None and height is not None:
        size = (width, height)
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import Draw, rdDepictor
        rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None

def draw_2d_svg(smiles: str, size: tuple[int, int] = (400, 300), width: Optional[int] = None, height: Optional[int] = None):
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
    except Exception:
        pass
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None

generate_2d_svg = draw_2d_svg

def generate_3d_conformer(smiles: str) -> tuple[Optional[str], float, float]:
    """
    Generates a 3D conformer using ETKDGv3, optimizes it using MMFF94 force field,
    and calculates Gasteiger partial charges.
    Returns (mol_block, min_charge, max_charge).
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, 0.0, 0.0
    
    try:
        # Add hydrogens for proper 3D geometry
        mol_3d = Chem.AddHs(mol)
        
        # Generate 3D coordinates using ETKDGv3
        embed_status = rdDistGeom.EmbedMolecule(mol_3d, rdDistGeom.ETKDGv3())
        if embed_status != 0:
            # Fallback to standard distance geometry if ETKDGv3 fails
            embed_status = rdDistGeom.EmbedMolecule(mol_3d)
            
        if embed_status == 0:
            # Optimize structure using MMFF94 force field
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol_3d)
            
        # Compute Gasteiger partial charges
        rdPartialCharges.ComputeGasteigerCharges(mol_3d)
        
        # Extract charge bounds
        charges = []
        for atom in mol_3d.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                try:
                    c = float(atom.GetProp("_GasteigerCharge"))
                    if not np.isnan(c) and not np.isinf(c):
                        charges.append(c)
                except ValueError:
                    pass
        
        min_charge = min(charges) if charges else 0.0
        max_charge = max(charges) if charges else 0.0
        
        # Convert to Mol block string
        mol_block = Chem.MolToMolBlock(mol_3d)
        return mol_block, min_charge, max_charge
    except Exception:
        # Fallback: compute Gasteiger charges on 2D mol if 3D conformer fails
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                if atom.HasProp("_GasteigerCharge"):
                    c = float(atom.GetProp("_GasteigerCharge"))
                    if not np.isnan(c) and not np.isinf(c):
                        charges.append(c)
            min_charge = min(charges) if charges else 0.0
            max_charge = max(charges) if charges else 0.0
            return None, min_charge, max_charge
        except Exception:
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
    matches = []
    for entry in catalog.GetMatches(mol):
        matches.append(entry.GetDescription())
    return matches

def qed_profile(smiles: str) -> Optional[dict]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return {
        "QED": round(QED.qed(mol), 4),
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotB": int(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": int(Lipinski.NumAromaticRings(mol)),
        "TPSA": round(Descriptors.TPSA(mol), 2),
    }

@lru_cache(maxsize=1)
def _load_train_fps():
    path = PROCESSED_DATA_DIR / "train_fps.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

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

@lru_cache(maxsize=1)
def _load_smiles_to_pdb() -> dict:
    path = PROCESSED_DATA_DIR / "smiles_to_pdb.json"
    if not path.exists():
        return {"known_ligands": {}}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"known_ligands": {}}


def _smiles_hash(smiles: str) -> str:
    import hashlib
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:12]


def lookup_pdb_ids(smiles: str) -> list[dict]:
    canon = canonicalize(smiles)
    if canon is None:
        return []
    try:
        from src.pdb_utils import find_gpcrdb_structure_matches
        matches = find_gpcrdb_structure_matches(canon, min_tanimoto=0.40)
        hits = []
        for m in matches:
            hits.append({
                "pdb_id": m["pdb_id"],
                "subtype": m["subtype"],
                "state": m["state"],
                "name": f"Human {m['subtype']} ({m['state']}) · {m['ligand_name']}",
                "url": m["rcsb_url"],
                "gpcrdb_url": m["gpcrdb_url"],
                "tanimoto": m["tanimoto"],
            })
        return hits
    except Exception:
        return []


def topk_tanimoto_with_pdb(smiles: str, k: int = 5) -> tuple[Optional[str], list[dict]]:
    from src.pdb_utils import real_structure_refs_with_analogs
    canon, top = topk_tanimoto(smiles, k=k)
    if not top:
        return canon, []
    results = []
    for smi, tan in top:
        results.append({
            "smiles": smi,
            "tanimoto": tan,
            "real_structures": real_structure_refs_with_analogs(smi),
        })
    return canon, results


@lru_cache(maxsize=1)
def _load_train_smiles() -> list[str]:
    path = PROCESSED_DATA_DIR / "train_smiles.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

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
    """Generates a 3D conformer and returns the PDB block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        mol_3d = Chem.AddHs(mol)
        embed_status = rdDistGeom.EmbedMolecule(mol_3d, rdDistGeom.ETKDGv3())
        if embed_status != 0:
            embed_status = rdDistGeom.EmbedMolecule(mol_3d)
        if embed_status == 0:
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol_3d)
        return Chem.MolToPDBBlock(mol_3d)
    except Exception:
        return None

def generate_sdf_block(smiles: str) -> Optional[str]:
    """Generates a 3D conformer and returns the SDF block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        mol_3d = Chem.AddHs(mol)
        embed_status = rdDistGeom.EmbedMolecule(mol_3d, rdDistGeom.ETKDGv3())
        if embed_status != 0:
            embed_status = rdDistGeom.EmbedMolecule(mol_3d)
        if embed_status == 0:
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol_3d)
        return Chem.MolToMolBlock(mol_3d)
    except Exception:
        return None

def generate_2d_mol_block(smiles: str) -> Optional[str]:
    """Generates a 2D conformer and returns the Mol block string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None