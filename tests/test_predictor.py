import logging
import warnings

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.config import SUBTYPES
from src.models import model_loader
from src.models.model_loader import (
    _load_lgb_models,
    _load_rf_models,
    _load_scaler,
    _load_xgb_models,
    clear_model_caches,
)
from src.predictor import _ensemble_predict, predict


class TestEnsemblePredict:
    def test_single_model(self):
        rng = np.random.default_rng(42)
        model = Ridge(alpha=1.0)
        x_train = rng.random((20, 5))
        y_train = rng.random(20)
        model.fit(x_train, y_train)
        x = rng.random(5)
        mean, std, low, high = _ensemble_predict([model], x)
        assert isinstance(mean, float)
        assert std >= 0.0
        assert low <= mean <= high


class TestModelLoaderContract:
    def test_legacy_stack_loader_removed_and_rf_loader_present(self):
        import src.predictor as predictor_mod

        assert hasattr(model_loader, "_load_rf_models")
        assert not hasattr(model_loader, "_load_stack_models")
        assert not hasattr(predictor_mod, "_load_stack_models")

    def test_production_models_and_scaler_load_without_warnings(self, caplog):
        clear_model_caches()
        with caplog.at_level(logging.WARNING):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                xgb_models = _load_xgb_models()
                lgb_models = _load_lgb_models()
                rf_models = _load_rf_models()
                scaler = _load_scaler("precise")

        assert len(caught) == 0, f"Unexpected warnings during model loading: {[str(w.message) for w in caught]}"
        warn_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warn_logs) == 0, f"Unexpected warning logs: {[r.getMessage() for r in warn_logs]}"

        assert set(xgb_models.keys()) == set(SUBTYPES)
        assert set(lgb_models.keys()) == set(SUBTYPES)
        assert set(rf_models.keys()) == set(SUBTYPES)
        assert hasattr(scaler, "transform")
        for st in SUBTYPES:
            st_scaler = _load_scaler("precise", subtype=st)
            assert hasattr(st_scaler, "transform")


class TestPredictContract:
    def test_predict_returns_xgb_lgb_rf_consensus_across_all_subtypes(self):
        result = predict("Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12")
        for model_key in ("XGBoost", "LightGBM", "RandomForest", "Consensus"):
            assert model_key in result["predictions"]
            assert model_key in result["uncertainty"]
            assert model_key in result["intervals"]
            for st in SUBTYPES:
                pred = result["predictions"][model_key][st]
                iv = result["intervals"][model_key][st]
                assert np.isfinite(pred)
                assert iv["lower"] <= pred <= iv["upper"]
                assert iv["width"] >= 0.0

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            predict("INVALID_SMILES_XYZ")
