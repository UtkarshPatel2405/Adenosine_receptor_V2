import importlib

import pytest

from src.run_pipeline import run_step


class TestRunStep:
    def test_successful_command(self):
        rc = run_step(
            ["-c", "print('hello')"],
            "Test command",
            allow_failure=False,
        )
        assert rc == 0

    def test_failed_command_allowed(self):
        rc = run_step(
            ["-c", "import sys; sys.exit(1)"],
            "Failing test",
            allow_failure=True,
        )
        assert rc == 1


class TestConfigPaths:
    def test_required_project_directories_exist(self):
        from src.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

        assert RAW_DATA_DIR.is_dir()
        assert PROCESSED_DATA_DIR.is_dir()
        assert MODELS_DIR.is_dir()
        assert OUTPUTS_DIR.is_dir()

    def test_subtypes_constant_matches_expected_order(self):
        from src.config import SUBTYPES

        assert SUBTYPES == ["A1", "A2A", "A2B", "A3"]
        assert len(SUBTYPES) == 4


class TestPipelineModuleImports:
    @pytest.mark.parametrize(
        "module_name,expected_attr",
        [
            ("src.run_pipeline", "main"),
            ("src.retrain_production", "train_conformal_model"),
            ("src.evaluator", "save_with_run_id"),
            ("src.shap_analysis", "_extract_xgb_estimator"),
            ("src.y_randomization", "run_y_randomization"),
            ("src.analysis", "shap_analysis"),
            ("src.fetch_chembl", "TARGETS"),
        ],
    )
    def test_core_module_imports_cleanly(self, module_name, expected_attr):
        mod = importlib.import_module(module_name)
        assert hasattr(mod, expected_attr), f"{module_name} missing expected symbol {expected_attr}"
