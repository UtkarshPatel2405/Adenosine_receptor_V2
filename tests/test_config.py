import os
from pathlib import Path

from src.config import (
    SUBTYPES, VALID_STANDARD_TYPES, DECOY_PCHEMBL,
    SCAFFOLD_TEST_SIZE, MAPIE_CONFIDENCE, LOG_LEVEL,
)


class TestConfigConstants:
    def test_subtypes_are_correct(self):
        assert SUBTYPES == ["A1", "A2A", "A2B", "A3"]

    def test_valid_standard_types(self):
        assert "KI" in VALID_STANDARD_TYPES
        assert "IC50" in VALID_STANDARD_TYPES
        assert "INVALID" not in VALID_STANDARD_TYPES

    def test_decoy_pchembl_is_physiologically_plausible(self):
        assert 3.0 <= DECOY_PCHEMBL <= 5.0

    def test_scaffold_test_size_is_reasonable(self):
        assert 0.1 <= SCAFFOLD_TEST_SIZE <= 0.3

    def test_mapie_confidence_is_standard(self):
        assert MAPIE_CONFIDENCE == 0.90

    def test_log_level_default(self):
        assert LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def test_env_var_override(self):
        os.environ["ADENOSINE_LOG_LEVEL"] = "DEBUG"
        from importlib import reload
        import src.config
        reload(src.config)
        assert src.config.LOG_LEVEL == "DEBUG"
        os.environ["ADENOSINE_LOG_LEVEL"] = "INFO"
        reload(src.config)
