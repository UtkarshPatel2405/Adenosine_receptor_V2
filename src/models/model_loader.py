"""Model and scaler caching loaders."""
from functools import lru_cache
import json
import logging
from pathlib import Path
import pickle
from typing import Dict, Any, Optional

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SUBTYPES

logger = logging.getLogger(__name__)


def _load_model_dict(prefix: str, subpath: str = "precise") -> Dict[str, Any]:
    """Helper to safely load models by prefix across subtypes, ignoring invalid pointers."""
    models: Dict[str, Any] = {}
    for st in SUBTYPES:
        for folder in (MODELS_DIR / subpath, MODELS_DIR):
            p = folder / f"{prefix}_{st}_production.pkl"
            if p.exists() and p.stat().st_size > 500:
                try:
                    with open(p, "rb") as f:
                        models[st] = pickle.load(f)
                    break
                except Exception as e:
                    logger.debug("Failed loading model %s: %s", p, e)
    return models


@lru_cache(maxsize=4)
def _load_scaler(mode: str = "precise"):
    """Load the feature standard scaler with multiple fallback search paths."""
    candidates = [
        MODELS_DIR / mode / f"scaler_{mode}.pkl",
        MODELS_DIR / "precise" / "scaler_precise.pkl",
        MODELS_DIR / "scaler.pkl",
        PROCESSED_DATA_DIR / "scaler.pkl",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 200:
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug("Scaler candidate %s failed: %s", p, e)
    raise FileNotFoundError("Valid production scaler not found in models/ directory.")


@lru_cache(maxsize=1)
def _load_xgb_models() -> Dict[str, Any]:
    return _load_model_dict("xgboost")


@lru_cache(maxsize=1)
def _load_lgb_models() -> Dict[str, Any]:
    return _load_model_dict("lgb")


@lru_cache(maxsize=1)
def _load_rf_models() -> Dict[str, Any]:
    return _load_model_dict("rf")


@lru_cache(maxsize=1)
def _load_stack_models() -> Dict[str, Any]:
    return _load_model_dict("stack_ridge")


@lru_cache(maxsize=1)
def _load_db_lookup() -> Dict[str, Any]:
    """Load curated bioactivity database lookup dictionary."""
    p = PROCESSED_DATA_DIR / "db_lookup.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
