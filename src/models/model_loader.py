"""Cached joblib/JSON loaders for production conformal models, scaler, and lookups."""
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SUBTYPES

logger = logging.getLogger(__name__)


def _load_model_dict(prefix: str, folder: str = "precise") -> Dict[str, Any]:
    """Load conformal production models by prefix across subtypes using joblib."""
    models: Dict[str, Any] = {}
    target_dir = MODELS_DIR / folder
    for st in SUBTYPES:
        p = target_dir / f"{prefix}_{st}_production.pkl"
        if p.exists() and p.stat().st_size > 500:
            try:
                models[st] = joblib.load(p)
            except Exception as exc:
                logger.warning("Failed loading model %s: %s", p, exc)
    return models


@lru_cache(maxsize=16)
def _load_scaler(mode: str = "precise", subtype: str | None = None):
    """Load per-subtype FeaturePipeline scaler (scaler_{subtype}.pkl) or fallback to scaler_precise.pkl."""
    folder = "functional" if mode == "functional" else "precise"
    base_dir = MODELS_DIR / folder
    if subtype is not None:
        st_path = base_dir / f"scaler_{subtype}.pkl"
        if st_path.exists() and st_path.stat().st_size > 200:
            try:
                return joblib.load(st_path)
            except Exception as exc:
                logger.warning("Failed loading subtype scaler %s: %s", st_path, exc)
    p = MODELS_DIR / "precise" / "scaler_precise.pkl"
    if p.exists() and p.stat().st_size > 200:
        try:
            return joblib.load(p)
        except Exception as exc:
            logger.warning("Failed loading scaler %s: %s", p, exc)
    raise FileNotFoundError(f"Production scaler not found at {p}.")


@lru_cache(maxsize=4)
def _load_consensus_weights(folder: str = "precise") -> Dict[str, Dict[str, float]]:
    """Load validation-weighted consensus weights per subtype (fallback to equal weights)."""
    p = MODELS_DIR / folder / "consensus_weights.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed loading consensus weights %s: %s", p, exc)
    return {st: {"XGBoost": 1 / 3, "LightGBM": 1 / 3, "RandomForest": 1 / 3} for st in SUBTYPES}


@lru_cache(maxsize=2)
def _load_xgb_models(folder: str = "precise") -> Dict[str, Any]:
    return _load_model_dict("xgboost", folder=folder)


@lru_cache(maxsize=2)
def _load_lgb_models(folder: str = "precise") -> Dict[str, Any]:
    return _load_model_dict("lgb", folder=folder)


@lru_cache(maxsize=2)
def _load_rf_models(folder: str = "precise") -> Dict[str, Any]:
    return _load_model_dict("rf", folder=folder)


def _load_json_lookup(filename: str) -> Dict[str, Any]:
    p = PROCESSED_DATA_DIR / filename
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed loading lookup %s: %s", p, exc)
        return {}


@lru_cache(maxsize=1)
def _load_db_lookup() -> Dict[str, Any]:
    """Load full curated bioactivity lookup dictionary."""
    return _load_json_lookup("db_lookup.json")


@lru_cache(maxsize=1)
def _load_db_lookup_train() -> Dict[str, Any]:
    """Load training-split bioactivity lookup dictionary."""
    return _load_json_lookup("db_lookup_train.json")


@lru_cache(maxsize=1)
def _load_db_lookup_functional() -> Dict[str, Any]:
    """Load full curated functional (pIC50/pEC50) lookup dictionary."""
    return _load_json_lookup("db_lookup_functional.json")


@lru_cache(maxsize=1)
def _load_db_lookup_functional_train() -> Dict[str, Any]:
    """Load training-split functional (pIC50/pEC50) lookup dictionary."""
    return _load_json_lookup("db_lookup_functional_train.json")


def clear_model_caches() -> None:
    """Clear all LRU model, scaler, and lookup caches."""
    _load_scaler.cache_clear()
    _load_consensus_weights.cache_clear()
    _load_xgb_models.cache_clear()
    _load_lgb_models.cache_clear()
    _load_rf_models.cache_clear()
    _load_db_lookup.cache_clear()
    _load_db_lookup_train.cache_clear()
    _load_db_lookup_functional.cache_clear()
    _load_db_lookup_functional_train.cache_clear()


def get_model_status() -> Dict[str, Any]:
    """Report on-disk availability of production models, per-subtype scalers, and lookups."""
    precise_dir = MODELS_DIR / "precise"
    scaler_path = precise_dir / "scaler_precise.pkl"
    subtype_scalers = [st for st in SUBTYPES if (precise_dir / f"scaler_{st}.pkl").exists()]
    xgb_present = [st for st in SUBTYPES if (precise_dir / f"xgboost_{st}_production.pkl").exists()]
    lgb_present = [st for st in SUBTYPES if (precise_dir / f"lgb_{st}_production.pkl").exists()]
    rf_present = [st for st in SUBTYPES if (precise_dir / f"rf_{st}_production.pkl").exists()]
    return {
        "scaler_precise": scaler_path.exists(),
        "subtype_scalers": subtype_scalers,
        "xgboost_subtypes": xgb_present,
        "lightgbm_subtypes": lgb_present,
        "random_forest_subtypes": rf_present,
        "db_lookup": (PROCESSED_DATA_DIR / "db_lookup.json").exists(),
        "db_lookup_train": (PROCESSED_DATA_DIR / "db_lookup_train.json").exists(),
    }
