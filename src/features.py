import logging
import pickle
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_CORR_THRESHOLD,
    FEATURE_NAN_THRESHOLD,
    FEATURE_VAR_THRESHOLD,
    MODELS_DIR,
    MORGAN_BITS,
    MORGAN_RADIUS,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


class FeatureFilter:
    """Filter molecular descriptors by missingness, variance, and pairwise correlation."""

    def __init__(self, nan_threshold=0.05, var_threshold=0.01, corr_threshold=0.90):
        self.nan_threshold = nan_threshold
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.selected_indices = None
        self.medians = None
        self.feature_names = None

    def fit(self, X: np.ndarray, feature_names=None):
        _, n_dims = X.shape
        nan_fraction = np.isnan(X).mean(axis=0)
        keep_nan_mask = nan_fraction <= self.nan_threshold

        self.medians = np.zeros(n_dims)
        for j in range(n_dims):
            col = X[:, j]
            valid_vals = col[~np.isnan(col)]
            self.medians[j] = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0

        X_filled = X.copy()
        for j in range(n_dims):
            nan_mask = np.isnan(X_filled[:, j])
            X_filled[nan_mask, j] = self.medians[j]

        variances = np.var(X_filled, axis=0)
        keep_var_mask = (variances >= self.var_threshold) & keep_nan_mask
        indices_after_var = np.where(keep_var_mask)[0]

        if len(indices_after_var) == 0:
            raise ValueError("All descriptors were filtered out by NaN/Variance thresholds.")

        X_filtered = X_filled[:, indices_after_var]
        n_features = X_filtered.shape[1]
        df_corr = pd.DataFrame(X_filtered).corr().abs()
        vars_filtered = variances[indices_after_var]

        to_drop = set()
        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if df_corr.iloc[i, j] > self.corr_threshold:
                    if vars_filtered[i] >= vars_filtered[j]:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)

        keep_indices = [indices_after_var[i] for i in range(n_features) if i not in to_drop]
        self.selected_indices = np.array(keep_indices, dtype=int)

        if feature_names is not None:
            self.feature_names = [feature_names[i] for i in self.selected_indices]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_transformed = X.copy()
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(1, -1)

        for j in range(X_transformed.shape[1]):
            nan_mask = np.isnan(X_transformed[:, j])
            X_transformed[nan_mask, j] = self.medians[j]

        return X_transformed[:, self.selected_indices]


class FeaturePipeline:
    """Combined descriptor filter and StandardScaler transform."""

    def __init__(self, feature_filter: FeatureFilter, scaler: StandardScaler):
        self.feature_filter = feature_filter
        self.scaler = scaler

    def transform(self, X_desc: np.ndarray) -> np.ndarray:
        X_desc_filtered = self.feature_filter.transform(X_desc)
        return self.scaler.transform(X_desc_filtered)


@lru_cache(maxsize=2048)
def _morgan_bits(smiles: str, n_bits: int = MORGAN_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    gen = _MORGAN_GEN if n_bits == MORGAN_BITS else rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=n_bits
    )
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.copy()


@lru_cache(maxsize=2048)
def _maccs_bits(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.copy()


CURATED_DESCRIPTORS_LIST = [
    "MolWt", "ExactMolWt", "HeavyAtomCount", "HeavyAtomMolWt",
    "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "FractionCSP3",
    "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge",
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
    "NumHeteroatoms", "NumValenceElectrons",
    "BalabanJ", "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0n", "Chi0v", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
    "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11",
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex",
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5",
    "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10",
    "SlogP_VSA11", "SlogP_VSA12",
    "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
    "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10",
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
    "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10",
    "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14",
    "NHOHCount", "NOCount",
    "NumRadicalElectrons",
    "qed",
]

_DESC_FUNCS = {}
for _name in CURATED_DESCRIPTORS_LIST:
    _func = getattr(Descriptors, _name, None) or getattr(Lipinski, _name, None)
    if _func is None and _name == "qed":
        try:
            from rdkit.Chem.QED import qed as _qed_func
            _func = _qed_func
        except ImportError:
            _func = None
    _DESC_FUNCS[_name] = _func


@lru_cache(maxsize=2048)
def _all_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    vals = []
    for name in CURATED_DESCRIPTORS_LIST:
        func = _DESC_FUNCS.get(name)
        if func is not None:
            try:
                vals.append(float(func(mol)))
            except Exception as exc:
                logger.warning("Descriptor %s failed for %s (%s); using NaN", name, smiles, exc)
                vals.append(np.nan)
        else:
            logger.warning("Descriptor %s unavailable in RDKit; using NaN", name)
            vals.append(np.nan)
    return np.array(vals, dtype=np.float32)


def _featurize_unique_smiles(unique_smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed

    Xfp = np.vstack(Parallel(n_jobs=-1)(delayed(_morgan_bits)(s) for s in unique_smiles))
    Xmaccs = np.vstack(Parallel(n_jobs=-1)(delayed(_maccs_bits)(s) for s in unique_smiles))
    Xdesc = np.vstack(Parallel(n_jobs=-1)(delayed(_all_descriptors)(s) for s in unique_smiles))
    return Xfp, Xmaccs, Xdesc


def _morgan_bitvect(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return _MORGAN_GEN.GetFingerprint(mol) if mol is not None else None


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    save_to_disk: bool = True,
    scaler_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, FeaturePipeline]:
    """Compute Morgan, MACCS, and filtered/scaled RDKit descriptors for train and test splits."""
    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_df[smiles_col].tolist()

    unique_train = list(dict.fromkeys(train_smiles))
    unique_test = list(dict.fromkeys(test_smiles))

    logger.info(
        "Featurizing %d train rows (%d unique) and %d test rows (%d unique)...",
        len(train_smiles),
        len(unique_train),
        len(test_smiles),
        len(unique_test),
    )

    u_fp_tr, u_maccs_tr, u_desc_tr = _featurize_unique_smiles(unique_train)
    train_idx_map = {s: i for i, s in enumerate(unique_train)}
    tr_idx = [train_idx_map[s] for s in train_smiles]
    X_fp_train = np.hstack([u_fp_tr[tr_idx], u_maccs_tr[tr_idx]])
    Xdesc_train = u_desc_tr[tr_idx]

    if unique_test:
        u_fp_te, u_maccs_te, u_desc_te = _featurize_unique_smiles(unique_test)
        test_idx_map = {s: i for i, s in enumerate(unique_test)}
        te_idx = [test_idx_map[s] for s in test_smiles]
        X_fp_test = np.hstack([u_fp_te[te_idx], u_maccs_te[te_idx]])
        Xdesc_test = u_desc_te[te_idx]
    else:
        X_fp_test = np.empty((0, X_fp_train.shape[1]), dtype=np.uint8)
        Xdesc_test = np.empty((0, Xdesc_train.shape[1]), dtype=np.float32)

    feature_filter = FeatureFilter(
        nan_threshold=FEATURE_NAN_THRESHOLD,
        var_threshold=FEATURE_VAR_THRESHOLD,
        corr_threshold=FEATURE_CORR_THRESHOLD,
    )
    feature_filter.fit(Xdesc_train, feature_names=list(CURATED_DESCRIPTORS_LIST))

    Xdesc_train_filtered = feature_filter.transform(Xdesc_train)
    logger.info(
        "Descriptor filter: %d -> %d descriptors retained.",
        Xdesc_train.shape[1],
        Xdesc_train_filtered.shape[1],
    )

    scaler = StandardScaler()
    Xdesc_train_s = scaler.fit_transform(Xdesc_train_filtered)

    if len(Xdesc_test) > 0:
        Xdesc_test_s = scaler.transform(feature_filter.transform(Xdesc_test))
    else:
        Xdesc_test_s = np.empty((0, Xdesc_train_s.shape[1]), dtype=np.float32)

    pipeline = FeaturePipeline(feature_filter, scaler)
    X_train = np.hstack([X_fp_train, Xdesc_train_s]).astype(np.float32)
    X_test = np.hstack([X_fp_test, Xdesc_test_s]).astype(np.float32)

    if save_to_disk:
        target_scaler = Path(scaler_path) if scaler_path is not None else MODELS_DIR / "precise" / "scaler_precise.pkl"
        target_scaler.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target_scaler, compress=3)

        train_fps = [_morgan_bitvect(s) for s in unique_train]
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR / "train_fps.pkl", "wb") as f:
            pickle.dump(train_fps, f)
        with open(PROCESSED_DATA_DIR / "train_smiles.pkl", "wb") as f:
            pickle.dump(unique_train, f)

    return X_train, X_test, pipeline


def build_features(smiles: str, pipeline: FeaturePipeline) -> np.ndarray:
    """Featurize a single SMILES using a fitted FeaturePipeline."""
    fp = _morgan_bits(smiles)
    maccs = _maccs_bits(smiles)
    desc = _all_descriptors(smiles).reshape(1, -1)
    desc_s = pipeline.transform(desc).ravel()
    return np.hstack([fp.astype(np.float32), maccs.astype(np.float32), desc_s.astype(np.float32)])
