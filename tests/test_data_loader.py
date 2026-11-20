import json
import tempfile

import pytest

from src.config import PROCESSED_DATA_DIR, SUBTYPES
from src.data_loader import SUBTYPE_MAP, _canonicalize_smiles, load_all_raw_data, load_and_clean


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
        assert SUBTYPE_MAP["A2BR"] == "A2B"
        assert SUBTYPE_MAP["A3R"] == "A3"

    def test_identity_mapping(self):
        for sub in ["A1", "A2A", "A2B", "A3"]:
            assert SUBTYPE_MAP[sub] == sub

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            _ = SUBTYPE_MAP["UNKNOWN"]


class TestLoadAllRawData:
    def test_nonexistent_directory_raises(self):
        with pytest.raises(ValueError, match="No valid.*datasets found"):
            load_all_raw_data("/nonexistent/path")

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No valid.*datasets found"):
                load_all_raw_data(tmpdir)

    def test_chembl_csv_loads_if_present(self, tmp_path):
        csv_path = tmp_path / "AR_all_unique_parents_with_smiles.csv"
        csv_path.write_text(
            "smiles,pchembl_value,standard_type,TAG,standard_relation,assay_type,confidence_score\n"
            "CCO,7.0,Ki,A1R,=,B,9\n"
            "CCC,6.5,IC50,A1R,=,B,9\n",
            encoding="utf-8",
        )
        df = load_all_raw_data(str(tmp_path))
        assert len(df) == 1
        assert df.iloc[0]["smiles"] == "CCO"

    def test_synthetic_primary_csv_deduplication_and_tiering(self, tmp_path):
        csv_path = tmp_path / "chembl_human_ar_bioactivities.csv"
        csv_path.write_text(
            "smiles,target_subtype,pchembl_value,standard_type,standard_relation,assay_type\n"
            "C(O)C,A1,7.0,Ki,=,B\n"
            "CCO,A1,8.0,Kd,=,B\n"
            "CCO,A1,5.2,IC50,=,B\n"
            "CCO,A2A,4.5,IC50,=,F\n"
            "CCCO,A2A,5.0,Ki,=,B\n"
            "CCCO,A2A,8.5,Ki,=,B\n"
            "c1ccccc1,A3,9.0,Ki,>,B\n"
            "c1ccccc1,A3,2.0,Ki,=,B\n"
            "INVALID,A1,7.5,Ki,=,B\n",
            encoding="utf-8",
        )
        lookup_out = tmp_path / "lookup.json"
        df, lookup = load_and_clean(
            data_dir=str(tmp_path),
            save_lookup_path=str(lookup_out),
            mode="precise",
            include_decoys=False,
        )

        assert lookup_out.exists()
        saved_lookup = json.loads(lookup_out.read_text(encoding="utf-8"))
        assert saved_lookup == lookup

        assert len(df) == 1
        assert lookup["CCO"]["A1"] == pytest.approx(7.5)
        assert "A2A" not in lookup["CCO"]
        assert "CCCO" not in lookup


class TestDecoyInjectionNoFalseNegatives:
    """Verify pure Ki/Kd loading, 4 subtype CSVs, weak-record retention, and absence of mutual decoys."""

    def test_primary_chembl_csv_loads_cleanly_without_mutating_processed(self, tmp_path):
        processed_lookup = PROCESSED_DATA_DIR / "db_lookup.json"
        mtime_before = processed_lookup.stat().st_mtime_ns if processed_lookup.exists() else None

        lookup_out = tmp_path / "lookup.json"
        df, lookup = load_and_clean(
            data_dir="data/raw",
            save_lookup_path=str(lookup_out),
            mode="precise",
            include_decoys=False,
        )

        assert lookup_out.exists()
        if mtime_before is not None:
            assert processed_lookup.stat().st_mtime_ns == mtime_before

        assert len(df) > 15000, f"Expected >15,000 deduplicated Ki/Kd rows, got {len(df)}"
        assert set(df["standard_type"].str.upper().unique()).issubset({"KI", "KD"})
        weak_count = int((df["pchembl_value"] < 6.0).sum())
        assert weak_count > 0, "Expected weak/inactive records (pKi < 6.0) to be retained"
        assert set(df["target_subtype"].unique()).issubset(set(SUBTYPES))

        for st in SUBTYPES:
            sub_csv = PROCESSED_DATA_DIR / f"{st}_pki.csv"
            assert sub_csv.is_file(), f"Missing subtype CSV: {sub_csv}"

        total_entries = sum(len(v) for v in lookup.values())
        avg_entries = total_entries / max(len(lookup), 1)
        assert 1.0 < avg_entries <= 2.5, (
            f"Expected >1 and <=2.5 average subtypes per compound, got {avg_entries:.3f}"
        )

    def test_known_targets_does_not_create_false_decoys(self, tmp_path):
        lookup_out = tmp_path / "lookup.json"
        _, lookup = load_and_clean(
            data_dir="data/raw",
            save_lookup_path=str(lookup_out),
            mode="precise",
            include_decoys=True,
        )

        assert lookup_out.exists()
        all_four_count = sum(
            1
            for subtypes in lookup.values()
            if all(subtypes.get(st) == 4.0 for st in SUBTYPES)
        )
        assert all_four_count <= 2000, (
            f"{all_four_count} compounds have pChEMBL=4.0 for ALL 4 subtypes — "
            "possible mutual decoy residue beyond expected P2Y decoys"
        )
