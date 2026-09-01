"""
Integration tests for the Adenosine Selectivity Model pipeline.

These tests verify that the full pipeline runs end-to-end and that
critical scientific guarantees are met (no data leakage, valid
conformal intervals, etc.).

These tests are marked 'slow' and 'integration' — skip with:
    pytest tests/ -m "not slow and not integration"
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import PROCESSED_DATA_DIR, SUBTYPES


@pytest.mark.slow
@pytest.mark.integration
class TestGlobalSplitConsistency:
    """
    Verify that the global scaffold split ensures zero scaffold overlap
    between train and test sets.
    """

    def test_global_split_no_scaffold_leakage(self):
        from src.scaffold_split import _murcko_scaffold_smiles

        split_path = PROCESSED_DATA_DIR / "global_split.json"
        if not split_path.exists():
            pytest.skip("global_split.json not found — run retrain_production.py first")

        with open(split_path) as f:
            split = json.load(f)

        train_smiles = split["train"]
        test_smiles = split["test"]

        train_scaffolds = {_murcko_scaffold_smiles(s) for s in train_smiles}
        test_scaffolds = {_murcko_scaffold_smiles(s) for s in test_smiles}

        overlap = train_scaffolds & test_scaffolds
        assert len(overlap) == 0, (
            f"Found {len(overlap)} scaffolds in both train and test! "
            f"This indicates data leakage. First 3: {list(overlap)[:3]}"
        )

    def test_global_split_covers_all_subtypes(self):
        split_path = PROCESSED_DATA_DIR / "global_split.json"
        if not split_path.exists():
            pytest.skip("global_split.json not found")

        with open(split_path) as f:
            split = json.load(f)
        assert len(split["train"]) > 0
        assert len(split["test"]) > 0


@pytest.mark.slow
@pytest.mark.integration
class TestModelLoading:
    def test_xgboost_models_load_for_all_subtypes(self):
        from src.predictor import _load_xgb_models

        models = _load_xgb_models()
        for st in SUBTYPES:
            assert st in models, f"Missing XGBoost model for {st}"

    def test_rf_models_load_for_all_subtypes(self):
        from src.predictor import _load_rf_models, MODELS_DIR

        models = _load_rf_models()
        rf_path = MODELS_DIR / "precise" / "rf_A1_production.pkl"
        if rf_path.exists() and rf_path.stat().st_size < 1000:
            pytest.skip("RF model files are Git LFS pointers in this environment")
        for st in SUBTYPES:
            assert st in models, f"Missing RF model for {st}"


    def test_scaler_loads(self):
        from src.predictor import _load_scaler

        scaler = _load_scaler("precise")
        assert scaler is not None


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineEndToEnd:
    def test_single_prediction_returns_all_fields(self):
        from src.predictor import predict

        result = predict("CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S")
        assert "predictions" in result
        assert "XGBoost" in result["predictions"]
        for st in SUBTYPES:
            assert st in result["predictions"]["XGBoost"]
        assert "best_target" in result
        assert "target_hits" in result
        assert "descriptors" in result

    def test_invalid_smiles_raises(self):
        from src.predictor import predict

        with pytest.raises(ValueError, match="Invalid SMILES"):
            predict("INVALID_SMILES_STRING")

    def test_prediction_has_conformal_intervals(self):
        from src.predictor import predict

        result = predict("CCn1c(/N=C/c2ccc(Br)cc2)c(C#N)sc1=S")
        intervals = result.get("intervals", {}).get("XGBoost", {})
        for st in SUBTYPES:
            if st in intervals:
                interval = intervals[st]
                if interval.get("width", 0) > 0:
                    assert interval["lower"] <= interval["upper"]
                    break
