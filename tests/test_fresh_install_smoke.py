"""Fresh-install smoke and regression tests for runtime environment, artifacts, presets, and SHAP."""
import math
from pathlib import Path

import pytest

from src.analysis import shap_analysis
from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ROOT_DIR, SUBTYPES
from src.predictor import predict
from src.presets import PRESETS

_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _version_tuple(version_str: str) -> tuple[int, ...]:
    parts = []
    for token in version_str.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


class TestFreshInstallEnvironment:
    def test_core_runtime_imports_and_versions(self):
        import fastapi
        import mapie
        import numpy as np
        import openpyxl
        import rdkit
        import scipy
        import shap
        import uvicorn
        import xgboost as xgb
        from mapie.regression import CrossConformalRegressor

        assert _version_tuple(np.__version__) >= (2, 0), f"numpy={np.__version__}"
        assert _version_tuple(xgb.__version__) >= (3, 0), f"xgboost={xgb.__version__}"
        assert _version_tuple(mapie.__version__) >= (1, 0), f"mapie={mapie.__version__}"
        assert CrossConformalRegressor is not None
        assert _version_tuple(fastapi.__version__) >= (0, 115), f"fastapi={fastapi.__version__}"
        assert uvicorn is not None
        assert openpyxl is not None
        assert scipy is not None
        assert rdkit is not None
        assert shap is not None

    def test_no_git_lfs_pointer_files_in_artifacts(self):
        lfs_pointers: list[str] = []
        for base_dir in (MODELS_DIR, DATA_DIR, OUTPUTS_DIR):
            assert base_dir.is_dir(), f"Required directory missing: {base_dir}"
            for path in sorted(base_dir.rglob("*")):
                if path.is_file():
                    with open(path, "rb") as f:
                        head = f.read(64)
                    if head.startswith(_LFS_HEADER):
                        lfs_pointers.append(path.relative_to(ROOT_DIR).as_posix())

        assert not lfs_pointers, (
            f"Found unhydrated Git LFS pointer files in models/data/outputs: {lfs_pointers}"
        )

    def test_single_markdown_file_is_readme(self):
        excluded_dirs = {".venv", ".git", ".pytest_cache"}
        md_files = [
            p.relative_to(ROOT_DIR).as_posix()
            for p in Path(ROOT_DIR).rglob("*.md")
            if not any(part in excluded_dirs for part in p.parts)
        ]
        assert md_files == ["README.md"], f"Expected only ['README.md'], found {md_files}"


@pytest.mark.integration
class TestPresetPredictionsAndShap:
    def test_all_eight_presets_predict_valid_affinities_and_conformal_intervals(self):
        expected_keys = {
            "Regadenoson (Lexiscan)",
            "CGS-21680",
            "Istradefylline",
            "PSB-603",
            "ZM241385",
            "CCPA",
            "BAY 60-6583",
            "IB-MECA (CF101)",
            "Caffeine",
            "Theophylline",
            "NECA",
            "DPCPX",
        }
        assert set(PRESETS.keys()) == expected_keys
        assert len(PRESETS) == 12

        for name, info in PRESETS.items():
            res = predict(info["smiles"])
            for model_name in ("XGBoost", "LightGBM", "RandomForest", "Consensus"):
                for st in SUBTYPES:
                    pred = res["predictions"][model_name][st]
                    iv = res["intervals"][model_name][st]
                    assert math.isfinite(pred), f"{name} {model_name} {st} non-finite: {pred}"
                    assert 3.0 <= pred <= 11.5, f"{name} {model_name} {st} out of bounds: {pred}"
                    assert iv["lower"] <= pred <= iv["upper"], (
                        f"Inverted/misaligned interval for {name} {model_name} {st}: pred={pred}, iv={iv}"
                    )
                    if model_name in ("XGBoost", "LightGBM", "Consensus"):
                        assert iv["width"] > 0.1, (
                            f"Degenerate interval width for {name} {model_name} {st}: iv={iv}"
                        )

    def test_zm241385_reference_affinity_and_database_status(self):
        zm_smiles = PRESETS["ZM241385"]["smiles"]
        res = predict(zm_smiles)
        assert res["predictions"]["Consensus"]["A2A"] >= 7.0
        assert res["in_database"] is True
        assert res["db_status"] in ("training", "test")

    def test_zm241385_shap_analysis_on_a2a(self):
        zm_smiles = PRESETS["ZM241385"]["smiles"]
        sh = shap_analysis(zm_smiles, "A2A")
        assert sh is not None
        assert sh["subtype"] == "A2A"
        assert math.isfinite(sh["base_value"])
        assert len(sh["features"]) > 0
        for feat in sh["features"]:
            assert "feature" in feat
            assert "value" in feat
            assert math.isfinite(feat["value"])
