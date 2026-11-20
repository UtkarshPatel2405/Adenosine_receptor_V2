"""
Integration tests for the Adenosine Receptor Profiler pipeline.

Verify scaffold split determinism and leakage prevention, production XGBoost/LightGBM
conformal model loading without warnings, and end-to-end prediction outputs.
"""
import json
import warnings

import numpy as np
import pytest

from src.config import (
    PROCESSED_DATA_DIR,
    SCAFFOLD_SPLIT_SEED,
    SCAFFOLD_TEST_SIZE,
    SUBTYPES,
)
from src.models.model_loader import (
    _load_lgb_models,
    _load_rf_models,
    _load_scaler,
    _load_xgb_models,
    clear_model_caches,
)
from src.predictor import predict
from src.scaffold_split import _murcko_scaffold_smiles, split_smiles_globally


@pytest.mark.slow
@pytest.mark.integration
class TestGlobalSplitConsistency:
    """Verify global scaffold split determinism and zero scaffold leakage."""

    def test_global_split_no_scaffold_leakage(self):
        split_path = PROCESSED_DATA_DIR / "global_split.json"
        assert split_path.is_file(), f"Missing {split_path}"

        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)

        train_smiles = split["train"]
        test_smiles = split["test"]
        assert set(train_smiles) & set(test_smiles) == set()

        train_scaffolds = {_murcko_scaffold_smiles(s) for s in train_smiles}
        test_scaffolds = {_murcko_scaffold_smiles(s) for s in test_smiles}

        overlap = train_scaffolds & test_scaffolds
        assert len(overlap) == 0, (
            f"Found {len(overlap)} scaffolds in both train and test. "
            f"First 3: {list(overlap)[:3]}"
        )

    def test_global_split_covers_nonempty_train_and_test(self):
        split_path = PROCESSED_DATA_DIR / "global_split.json"
        assert split_path.is_file(), f"Missing {split_path}"

        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        assert len(split["train"]) > 0
        assert len(split["test"]) > 0

    def test_global_split_is_deterministic_and_reproducible(self):
        split_path = PROCESSED_DATA_DIR / "global_split.json"
        assert split_path.is_file(), f"Missing {split_path}"

        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)

        all_smiles = split["train"] + split["test"]
        train_1, test_1 = split_smiles_globally(
            all_smiles,
            test_size=SCAFFOLD_TEST_SIZE,
            random_state=SCAFFOLD_SPLIT_SEED,
        )
        train_2, test_2 = split_smiles_globally(
            list(reversed(all_smiles)),
            test_size=SCAFFOLD_TEST_SIZE,
            random_state=SCAFFOLD_SPLIT_SEED,
        )

        assert train_1 == train_2
        assert test_1 == test_2
        assert train_1 == set(split["train"])
        assert test_1 == set(split["test"])


@pytest.mark.slow
@pytest.mark.integration
class TestModelLoading:
    def test_xgboost_models_load_for_all_subtypes(self):
        clear_model_caches()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = _load_xgb_models()
        assert not caught, f"Warnings loading XGBoost models: {[str(w.message) for w in caught]}"
        for st in SUBTYPES:
            assert st in models, f"Missing XGBoost model for {st}"

    def test_lightgbm_models_load_for_all_subtypes(self):
        clear_model_caches()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = _load_lgb_models()
        assert not caught, f"Warnings loading LightGBM models: {[str(w.message) for w in caught]}"
        for st in SUBTYPES:
            assert st in models, f"Missing LightGBM model for {st}"

    def test_random_forest_models_load_for_all_subtypes(self):
        clear_model_caches()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = _load_rf_models()
        assert not caught, f"Warnings loading RandomForest models: {[str(w.message) for w in caught]}"
        for st in SUBTYPES:
            assert st in models, f"Missing RandomForest model for {st}"

    def test_scaler_loads(self):
        clear_model_caches()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scaler = _load_scaler("precise")
        assert not caught, f"Warnings loading scaler: {[str(w.message) for w in caught]}"
        assert scaler is not None
        assert hasattr(scaler, "transform")


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineEndToEnd:
    def test_single_prediction_returns_all_fields(self):
        result = predict("Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12")
        expected_keys = {
            "predictions",
            "uncertainty",
            "intervals",
            "ki_values",
            "best_target",
            "target_hits",
            "selectivity_spectrum",
            "applicability_domain",
            "functional_efficacy",
            "pocket_interactions",
            "safety_profile",
            "cns_admet",
            "descriptors",
            "in_database",
            "db_status",
            "db_value",
            "source",
        }
        missing = expected_keys - set(result.keys())
        assert not missing, f"Missing keys in predict() result: {missing}"

        for model_name in ("XGBoost", "LightGBM", "RandomForest", "Consensus"):
            assert model_name in result["predictions"]
            for st in SUBTYPES:
                val = result["predictions"][model_name][st]
                assert np.isfinite(val)

        assert result["best_target"] in SUBTYPES
        assert result["db_status"] in ("training", "test", "de_novo")

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            predict("INVALID_SMILES_STRING")

    def test_prediction_has_conformal_intervals(self):
        result = predict("Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12")
        for model_name in ("XGBoost", "LightGBM", "Consensus"):
            intervals = result["intervals"][model_name]
            preds = result["predictions"][model_name]
            for st in SUBTYPES:
                iv = intervals[st]
                pred = preds[st]
                assert iv["lower"] <= pred <= iv["upper"]
                assert iv["width"] > 0.0

    def test_de_novo_molecule_status(self):
        result = predict("CC(=O)Oc1ccccc1C(=O)O")
        assert result["db_status"] == "de_novo"
        assert result["in_database"] is False
        assert result["db_value"] is None
        assert result["source"] == "model"
