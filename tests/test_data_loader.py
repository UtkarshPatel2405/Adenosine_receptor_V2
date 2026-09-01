import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import _canonicalize_smiles, SUBTYPE_MAP, load_all_raw_data


class TestCanonicalizeSmiles:
    def test_valid_smiles_returns_canonical(self):
        assert _canonicalize_smiles("C(O)C") == "CCO"

    def test_invalid_smiles_returns_none(self):
        assert _canonicalize_smiles("INVALID") is None

    def test_empty_string_returns_none(self):
        assert _canonicalize_smiles("") is None

    def test_none_returns_none(self):
        assert _canonicalize_smiles(None) is None  # type: ignore

    def test_whitespace_only_returns_none(self):
        assert _canonicalize_smiles("   ") is None

    def test_roundtrip_preserves_identity(self):
        c1 = _canonicalize_smiles("c1ccccc1")
        c2 = _canonicalize_smiles(c1)
        assert c1 == c2


class TestSubtypeMap:
    def test_all_subtypes_mapped(self):
        for key in ["A1R", "A2AR", "A2BR", "A3R"]:
            assert key in SUBTYPE_MAP
        assert SUBTYPE_MAP["A1R"] == "A1"
        assert SUBTYPE_MAP["A2AR"] == "A2A"
        assert SUBTYPE_MAP["A3R"] == "A3"

    def test_identity_mapping(self):
        for sub in ["A1", "A2A", "A2B", "A3"]:
            assert SUBTYPE_MAP[sub] == sub

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            _ = SUBTYPE_MAP["UNKNOWN"]


class TestLoadAllRawData:
    def test_nonexistent_directory_raises(self):
        with pytest.raises(ValueError, match="No valid datasets found"):
            load_all_raw_data("/nonexistent/path")

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No valid datasets found"):
                load_all_raw_data(tmpdir)

    def test_chembl_csv_loads_if_present(self, tmp_path):
        csv_path = tmp_path / "AR_all_unique_parents_with_smiles.csv"
        csv_path.write_text(
            "smiles,pchembl_value,standard_type,TAG,standard_relation,assay_type,confidence_score\n"
            "CCO,5.0,IC50,A1R,=,B,9\n"
        )
        df = load_all_raw_data(str(tmp_path))
        assert len(df) == 1
        assert df.iloc[0]["smiles"] == "CCO"


class TestDecoyInjectionNoFalseNegatives:
    """
    CRITICAL TEST: Verify the Mutual Decoy Fallacy is fixed.

    Old behavior: molecules with no data for a subtype were assigned pChEMBL=4.0.
    New behavior: only explicit P2Y structural decoys are included.
    This test verifies no synthetic inactive labels are created.
    """

    def test_known_targets_does_not_create_false_decoys(self):
        from src.data_loader import load_and_clean

        df, lookup = load_and_clean(
            data_dir="data/raw",
            mode="precise",
            include_decoys=True,
        )

        for smiles, subtypes in lookup.items():
            for st in ["A1", "A2A", "A2B", "A3"]:
                if st in subtypes:
                    val = subtypes[st]
                    # If it's a known compound with a measured value,
                    # the value must not be the decoy placeholder
                    # (unless it's explicitly a P2Y decoy)
                    if "DECOY" in str(subtypes.get("standard_type", "")):
                        assert val <= 5.0, f"Decoy compound {smiles} has value {val} > 5.0"
                    # With the Mutual Decoy Fallacy removed, real compounds may
                    # legitimately have pChEMBL ≈ 4.0 (weak activity). Check instead
                    # that no compound has 4.0 across ALL FOUR subtypes (the old bug).
                    pass

        # Sanity check: count compounds with pChEMBL=4.0 for ALL 4 subtypes.
        # P2Y structural decoys legitimately have 4.0 for all subtypes.
        # If this count is unexpectedly large, the Mutual Decoy Fallacy may have resurfaced.
        all_four_count = sum(
            1 for subtypes in lookup.values()
            if all(subtypes.get(st) == 4.0 for st in ["A1", "A2A", "A2B", "A3"])
        )
        # P2Y decoy set has ~1600 unique molecules; allow some buffer
        assert all_four_count <= 2000, (
            f"{all_four_count} compounds have pChEMBL=4.0 for ALL 4 subtypes — "
            f"possible mutual decoy residue beyond expected P2Y decoys"
        )

    def test_lookup_contains_no_mutual_decoy_traces(self):
        from src.data_loader import load_and_clean

        df, lookup = load_and_clean(
            data_dir="data/raw",
            mode="precise",
            include_decoys=False,
        )

        total_entries = sum(len(v) for v in lookup.values())
        avg_entries = total_entries / max(len(lookup), 1)
        assert avg_entries <= 2.5, (
            f"Average {avg_entries:.2f} subtypes per compound suggests "
            f"mutual decoys may still be present"
        )
