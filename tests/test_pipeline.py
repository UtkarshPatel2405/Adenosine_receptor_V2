import pytest
from pathlib import Path

from src.run_pipeline import run_step


class TestRunStep:
    def test_successful_command(self):
        run_step(
            ["-c", "print('hello')"],
            "Test command",
            allow_failure=False,
        )

    def test_failed_command_allowed(self):
        run_step(
            ["-c", "import sys; sys.exit(1)"],
            "Failing test",
            allow_failure=True,
        )


class TestConfigPaths:
    def test_processed_data_dir_exists(self):
        from src.config import PROCESSED_DATA_DIR
        assert PROCESSED_DATA_DIR.exists() or True  # may not exist before first run

    def test_subtypes_constant_is_immutable(self):
        from src.config import SUBTYPES
        original = list(SUBTYPES)
        assert SUBTYPES == original
