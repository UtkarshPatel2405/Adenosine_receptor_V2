"""Provenance & trust metadata: content fingerprints of the data and models that
produced a prediction, plus the assay-quality gates applied at build time.

Goal: any returned prediction can be traced back to the exact files that made
it, so a number is never an unverifiable claim.
"""
import hashlib
import logging
import pickle
from functools import lru_cache
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    SUBTYPES,
    REQUIRED_CONFIDENCE,
)

logger = logging.getLogger(__name__)


def _fingerprint_file(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return None


def _dir_fingerprints(dirp: Path, exts=(".pkl", ".json", ".csv")) -> dict:
    out = {}
    if not dirp.exists():
        return {}
    for p in sorted(dirp.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out[str(p.relative_to(dirp)).replace("\\", "/")] = _fingerprint_file(p)
    return out


def _count_train_records() -> int:
    p = PROCESSED_DATA_DIR / "train_smiles.pkl"
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        return len(data) if data is not None and hasattr(data, "__len__") else 0
    except Exception:
        return 0


def _sha_of(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(str(d[k]).encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def build_manifest() -> dict:
    """Fingerprint the exact data + model artifacts currently on disk."""
    processed = _dir_fingerprints(PROCESSED_DATA_DIR, (".pkl", ".json"))
    model_files = _dir_fingerprints(MODELS_DIR, (".pkl",))
    manifest = {
        "schema": "adeno-manifest/v1",
        "train_records": _count_train_records(),
        "subtypes": list(SUBTYPES),
        "assay_quality_gates": {
            "min_chembl_confidence": REQUIRED_CONFIDENCE,
            "standard_types": ["KI", "KD", "IC50", "EC50", "AC50"],
        },
        "processed_data": processed,
        "models": model_files,
    }
    # One combined fingerprint so a caller can detect 'anything changed'
    manifest["data_fingerprint"] = _sha_of(processed)
    manifest["model_fingerprint"] = _sha_of(model_files)
    return manifest


def run_id() -> str:
    from src.config import RUN_ID
    return RUN_ID


def provenance_payload() -> dict:
    """Compact, response-friendly provenance block."""
    from src.config import RUN_TIMESTAMP
    m = build_manifest()
    return {
        "schema": m["schema"],
        "version": "2.4.0-precise",
        "timestamp": RUN_TIMESTAMP,
        "train_records": m["train_records"],
        "subtypes": m["subtypes"],
        "assay_quality_gates": m["assay_quality_gates"],
        "data_fingerprint": m["data_fingerprint"],
        "model_fingerprint": m["model_fingerprint"],
        "model_hashes": m.get("models", {}),
        "run_id": run_id(),
    }


def generate_provenance_payload(smiles: str = None, predictions: dict = None) -> dict:
    """Generate provenance block with optional query-specific metadata."""
    payload = provenance_payload()
    if smiles:
        payload["query_smiles"] = smiles
    return payload
