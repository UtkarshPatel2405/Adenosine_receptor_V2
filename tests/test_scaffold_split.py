import pytest
import pandas as pd


class TestScaffoldSplit:
    def test_split_returns_dataframes(self):
        from src.scaffold_split import scaffold_split
        df = pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1", "CCN", "c1ccccc1O", "CCC", "C1CCCCC1"],
            "activity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        train_df, test_df = scaffold_split(df, test_size=0.3, smiles_col="smiles")
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(df)

    def test_missing_column_raises(self):
        from src.scaffold_split import scaffold_split
        df = pd.DataFrame({"smiles": ["CCO"], "activity": [1.0]})
        with pytest.raises(ValueError):
            scaffold_split(df, smiles_col="nonexistent")

    def test_split_smiles_globally(self):
        from src.scaffold_split import split_smiles_globally
        smiles = ["CCO", "c1ccccc1", "CCN", "c1ccccc1O", "CCC", "C1CCCCC1"]
        train_set, test_set = split_smiles_globally(smiles, test_size=0.3)
        assert len(train_set) > 0
        assert len(test_set) > 0
        assert len(train_set) + len(test_set) == len(set(smiles))
