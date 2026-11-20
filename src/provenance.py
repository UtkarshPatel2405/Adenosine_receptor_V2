"""Provenance and SHA-256 artifact integrity verification for inference outputs."""
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import (
    APP_VERSION,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    REQUIRED_CONFIDENCE,
    SUBTYPES,
    VALID_STANDARD_TYPES,
)

logger = logging.getLogger(__name__)


def _fingerprint_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception as exc:
        logger.warning("SHA-256 fingerprint failed for %s: %s", path, exc)
        return None


def _count_train_records() -> int:
    p = PROCESSED_DATA_DIR / "train_smiles.pkl"
    if not p.exists():
        return 0
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        return len(data) if data is not None and hasattr(data, "__len__") else 0
    except Exception as exc:
        logger.warning("Failed counting training records in %s: %s", p, exc)
        return 0


def _sha_of(d: Dict[str, Optional[str]]) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode("utf-8"))
        h.update(str(d[k]).encode("utf-8"))
    return h.hexdigest()[:16]


def _load_training_summary() -> Dict[str, Any]:
    summary_path = OUTPUTS_DIR / "training_summary.json"
    if not summary_path.exists():
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed reading %s: %s", summary_path, exc)
        return {}


def build_manifest() -> Dict[str, Any]:
    """Compute SHA-256 digests of active data splits, lookups, scaler, and production models."""
    data_artifacts = {
        "db_lookup.json": _fingerprint_file(PROCESSED_DATA_DIR / "db_lookup.json"),
        "db_lookup_train.json": _fingerprint_file(PROCESSED_DATA_DIR / "db_lookup_train.json"),
        "global_split.json": _fingerprint_file(PROCESSED_DATA_DIR / "global_split.json"),
    }

    model_artifacts: Dict[str, Optional[str]] = {}
    scaler_path = MODELS_DIR / "precise" / "scaler_precise.pkl"
    if scaler_path.exists():
        model_artifacts["precise/scaler_precise.pkl"] = _fingerprint_file(scaler_path)

    for st in SUBTYPES:
        st_scaler = MODELS_DIR / "precise" / f"scaler_{st}.pkl"
        if st_scaler.exists():
            model_artifacts[f"precise/scaler_{st}.pkl"] = _fingerprint_file(st_scaler)
        for prefix in ("xgboost", "lgb", "rf"):
            rel = f"precise/{prefix}_{st}_production.pkl"
            p = MODELS_DIR / rel
            if p.exists():
                model_artifacts[rel] = _fingerprint_file(p)
        func_scaler = MODELS_DIR / "functional" / f"scaler_{st}.pkl"
        if func_scaler.exists():
            model_artifacts[f"functional/scaler_{st}.pkl"] = _fingerprint_file(func_scaler)
        for prefix in ("xgboost", "lgb", "rf"):
            rel_f = f"functional/{prefix}_{st}_production.pkl"
            pf = MODELS_DIR / rel_f
            if pf.exists():
                model_artifacts[rel_f] = _fingerprint_file(pf)

    summary = _load_training_summary()
    return {
        "schema": "adeno-manifest/v2",
        "app_version": APP_VERSION,
        "training_run_id": summary.get("run_id"),
        "trained_at": summary.get("trained_at"),
        "dataset_source": summary.get("dataset_source"),
        "train_records": summary.get("n_train_molecules", _count_train_records()),
        "subtypes": list(SUBTYPES),
        "assay_quality_gates": {
            "min_chembl_confidence": REQUIRED_CONFIDENCE,
            "standard_types": sorted(VALID_STANDARD_TYPES),
        },
        "processed_data": data_artifacts,
        "models": model_artifacts,
        "data_fingerprint": _sha_of(data_artifacts),
        "model_fingerprint": _sha_of(model_artifacts),
    }


def run_id() -> str:
    summary = _load_training_summary()
    if summary.get("run_id"):
        return str(summary["run_id"])
    from src.config import RUN_ID
    return RUN_ID


def provenance_payload() -> Dict[str, Any]:
    """Build compact provenance block backed by training_summary.json and SHA-256 digests."""
    from src.config import RUN_ID, RUN_TIMESTAMP

    manifest = build_manifest()
    trained_id = manifest.get("training_run_id") or RUN_ID
    trained_at = manifest.get("trained_at") or RUN_TIMESTAMP

    return {
        "schema": manifest["schema"],
        "version": APP_VERSION,
        "app_version": APP_VERSION,
        "training_run_id": trained_id,
        "run_id": trained_id,
        "trained_at": trained_at,
        "timestamp": trained_at,
        "dataset_source": manifest.get("dataset_source") or "chembl_human_ar_bioactivities.csv",
        "train_records": manifest["train_records"],
        "subtypes": manifest["subtypes"],
        "assay_quality_gates": manifest["assay_quality_gates"],
        "data_fingerprint": manifest["data_fingerprint"],
        "model_fingerprint": manifest["model_fingerprint"],
        "data_hashes": manifest["processed_data"],
        "model_hashes": manifest["models"],
        "artifacts": {**manifest["processed_data"], **manifest["models"]},
    }


def generate_provenance_payload(smiles: Optional[str] = None, predictions: Optional[dict] = None) -> Dict[str, Any]:
    """Generate provenance block with optional query SMILES."""
    payload = provenance_payload()
    if smiles:
        payload["query_smiles"] = smiles
    return payload
