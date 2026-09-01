import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluator import _calibration_quartiles, save_with_run_id


class TestCalibrationQuartiles:
    def test_returns_4_bins_for_large_n(self):
        y_true = np.random.rand(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        y_std = np.abs(np.random.randn(100)) * 0.1
        quartiles = _calibration_quartiles(y_true, y_pred, y_std)
        assert len(quartiles) == 4

    def test_monotonically_increasing_mae(self):
        """
        A well-calibrated model should show increasing MAE across quartiles
        when y_std accurately reflects uncertainty.
        """
        np.random.seed(42)
        n = 200
        # Use a simple linear model with noise proportional to predicted std
        X = np.linspace(0, 1, n)
        y_true = 2 * X + 0.5
        # Noise scales with std: bins 0-3 have different noise levels
        # We sort by std, so lower std bins get low noise and higher std bins get high noise
        y_std = np.linspace(0.01, 0.8, n)
        np.random.shuffle(y_std)
        # Scale prediction noise by std
        y_pred = y_true + y_std * np.random.randn(n)

        quartiles = _calibration_quartiles(y_true, y_pred, y_std)
        maes = [q["mae_mean"] for q in quartiles]
        assert maes == sorted(maes), f"MAEs not monotonic: {maes}"

    def test_returns_empty_for_small_n(self):
        y_true = np.random.rand(4)
        y_pred = np.random.rand(4)
        y_std = np.random.rand(4)
        assert _calibration_quartiles(y_true, y_pred, y_std) == []

    def test_std_mean_tracks_uncertainty(self):
        np.random.seed(42)
        n = 40
        y_true = np.random.rand(n)
        y_pred = y_true.copy()
        y_std = np.concatenate([np.ones(10) * 0.01, np.ones(30) * 0.1])
        quartiles = _calibration_quartiles(y_true, y_pred, y_std)
        # First quartile should have lowest std_mean
        std_means = [q["std_mean"] for q in quartiles]
        assert std_means == sorted(std_means)


class TestSaveWithRunId:
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            save_with_run_id({"key": "value", "num": 42}, dirpath, "test", "RUN001")
            # Should create barcoded file and pointer
            barcoded = dirpath / "RUN001_test.json"
            pointer = dirpath / "test.json"
            assert barcoded.exists()
            assert pointer.exists()
            with open(barcoded) as f:
                data = json.load(f)
            assert data["key"] == "value"
            assert data["num"] == 42

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir) / "nested" / "deep"
            save_with_run_id({"a": 1}, dirpath, "test_nested", "RUN002")
            assert (dirpath / "RUN002_test_nested.json").exists()

    def test_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            save_with_run_id({"first": True}, dirpath, "overwrite_test", "RUN003")
            save_with_run_id({"second": True}, dirpath, "overwrite_test", "RUN004")
            # Pointer should point to newest
            with open(dirpath / "overwrite_test.json") as f:
                pointer = json.load(f)
            assert pointer["run_id"] == "RUN004"
