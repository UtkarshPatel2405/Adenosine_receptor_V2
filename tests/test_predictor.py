import pytest
import numpy as np


class TestEnsemblePredict:
    def test_single_model(self):
        from src.predictor import _ensemble_predict
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        model.fit(X, y)
        x = np.random.rand(5)
        mean, std, low, high = _ensemble_predict([model], x)
        assert isinstance(mean, float)
        assert std >= 0
        assert low <= mean <= high


class TestLoadScaler:
    def test_loads_scaler_with_fallback(self):
        from src.predictor import _load_scaler
        _load_scaler.cache_clear()
        # Even with a non-existent mode, _load_scaler falls back to models/scaler.pkl
        scaler = _load_scaler("nonexistent_mode_xyz")
        assert hasattr(scaler, 'transform'), "Scaler should have transform method"



