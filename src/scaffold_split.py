# src/scaffold_split.py
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def _murcko_scaffold_smiles(smiles: str) -> str:
    """Return Bemis Murcko scaffold SMILES (canonical, without chirality)."""
    try:
        scaf_smiles = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        if not scaf_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.RemoveStereochemistry(mol)
                return Chem.MolToSmiles(mol, canonical=True)
            return "__INVALID__"
        return scaf_smiles
    except Exception:
        return "__INVALID__"

def scaffold_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, smiles_col: str = "smiles"):
    """
    Split by scaffolds: whole scaffolds go to train or test.
    Default smiles_col is 'smiles' to match the ml_base.py pipeline.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Dataframe must contain the column: '{smiles_col}'")

    # Generate scaffolds for each molecule
    scaffolds = df[smiles_col].apply(_murcko_scaffold_smiles)
    df_copy = df.copy()
    df_copy["_scaffold"] = scaffolds

    # Group row indices by their scaffold type
    scaffold_to_indices = {}
    for i, scaf in enumerate(df_copy["_scaffold"].tolist()):
        scaffold_to_indices.setdefault(scaf, []).append(i)

    # Shuffle the unique scaffolds
    rng = random.Random(random_state)
    scaffold_keys = list(scaffold_to_indices.keys())
    rng.shuffle(scaffold_keys)

    n_total = len(df_copy)
    n_test_target = int(round(test_size * n_total))

    test_indices = []
    train_indices = []
    test_count = 0

    # Distribute scaffolds until the test set target size is reached
    for scaf in scaffold_keys:
        indices = scaffold_to_indices[scaf]
        if test_count < n_test_target:
            remaining = n_test_target - test_count
            if len(indices) <= remaining:
                test_indices.extend(indices)
                test_count += len(indices)
            else:
                # Probabilistic assignment to avoid overshoot: add to test
                # with probability proportional to remaining gap
                if rng.random() * n_test_target < remaining:
                    test_indices.extend(indices)
                    test_count += len(indices)
                else:
                    train_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # Create final dataframes
    train_df = df_copy.iloc[train_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    test_df = df_copy.iloc[test_indices].drop(columns=["_scaffold"]).reset_index(drop=True)

    return train_df, test_df


def split_smiles_globally(smiles_list: list, test_size: float = 0.2, random_state: int = 42) -> tuple[set, set]:
    """
    Split a list of unique SMILES globally by Bemis-Murcko scaffolds.
    Returns (train_smiles_set, test_smiles_set).
    """
    unique_smiles = list(set(smiles_list))
    
    scaffold_to_smiles = {}
    for smiles in unique_smiles:
        scaf = _murcko_scaffold_smiles(smiles)
        scaffold_to_smiles.setdefault(scaf, []).append(smiles)
        
    rng = random.Random(random_state)
    scaffold_keys = list(scaffold_to_smiles.keys())
    rng.shuffle(scaffold_keys)
    
    n_total = len(unique_smiles)
    n_test_target = int(round(test_size * n_total))
    
    test_smiles = set()
    train_smiles = set()
    test_count = 0
    
    for scaf in scaffold_keys:
        smi_list = scaffold_to_smiles[scaf]
        if test_count < n_test_target:
            remaining = n_test_target - test_count
            if len(smi_list) <= remaining:
                test_smiles.update(smi_list)
                test_count += len(smi_list)
            else:
                if rng.random() * n_test_target < remaining:
                    test_smiles.update(smi_list)
                    test_count += len(smi_list)
                else:
                    train_smiles.update(smi_list)
        else:
            train_smiles.update(smi_list)
            
    return train_smiles, test_smiles
