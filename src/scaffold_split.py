import logging
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def _murcko_scaffold_smiles(smiles: str) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES without chirality."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return "__INVALID__"
    try:
        scaf_smiles = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaf_smiles:
            mol_copy = Chem.Mol(mol)
            Chem.RemoveStereochemistry(mol_copy)
            return Chem.MolToSmiles(mol_copy, canonical=True)
        return scaf_smiles
    except Exception as exc:
        logger.warning("Murcko scaffold extraction failed for %r: %s", smiles, exc)
        return "__INVALID__"


def _partition_scaffolds(
    scaffold_keys: list[str],
    groups_map: dict[str, list],
    n_total: int,
    test_size: float,
    random_state: int,
) -> tuple[list, list]:
    rng = random.Random(random_state)
    ordered_keys = sorted(scaffold_keys)
    rng.shuffle(ordered_keys)

    n_test_target = int(round(test_size * n_total))
    train_items: list = []
    test_items: list = []
    test_count = 0

    for scaf in ordered_keys:
        items = groups_map[scaf]
        if test_count < n_test_target:
            remaining = n_test_target - test_count
            if len(items) <= remaining or (rng.random() * n_test_target < remaining):
                test_items.extend(items)
                test_count += len(items)
            else:
                train_items.extend(items)
        else:
            train_items.extend(items)

    return train_items, test_items


def scaffold_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    smiles_col: str = "smiles",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically partition a dataframe so whole Murcko scaffolds stay together."""
    if smiles_col not in df.columns:
        raise ValueError(f"Dataframe must contain the column: '{smiles_col}'")

    scaffolds = df[smiles_col].apply(_murcko_scaffold_smiles)
    df_copy = df.copy()
    df_copy["_scaffold"] = scaffolds

    scaffold_to_indices: dict[str, list[int]] = {}
    for i, scaf in enumerate(df_copy["_scaffold"].tolist()):
        scaffold_to_indices.setdefault(scaf, []).append(i)

    train_indices, test_indices = _partition_scaffolds(
        list(scaffold_to_indices.keys()),
        scaffold_to_indices,
        len(df_copy),
        test_size,
        random_state,
    )

    train_df = df_copy.iloc[train_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    test_df = df_copy.iloc[test_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    return train_df, test_df


def split_smiles_globally(
    smiles_list: list | pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[set[str], set[str]]:
    """Deterministically split unique SMILES globally by Bemis-Murcko scaffolds."""
    unique_smiles = sorted(set(smiles_list))
    scaffold_to_smiles: dict[str, list[str]] = {}
    for smiles in unique_smiles:
        scaf = _murcko_scaffold_smiles(smiles)
        scaffold_to_smiles.setdefault(scaf, []).append(smiles)

    train_items, test_items = _partition_scaffolds(
        list(scaffold_to_smiles.keys()),
        scaffold_to_smiles,
        len(unique_smiles),
        test_size,
        random_state,
    )
    return set(train_items), set(test_items)
