import logging
import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import rdFingerprintGenerator

from src.config import (
    FEATURE_NAN_THRESHOLD, FEATURE_VAR_THRESHOLD, FEATURE_CORR_THRESHOLD,
    MODELS_DIR, PROCESSED_DATA_DIR
)

logger = logging.getLogger(__name__)


class FeatureFilter:
    def __init__(self, nan_threshold=0.05, var_threshold=0.01, corr_threshold=0.90):
        self.nan_threshold = nan_threshold
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.selected_indices = None
        self.medians = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names=None):
        N, D = X.shape
        
        # 1. Calculate NaN fractions
        nan_fraction = np.isnan(X).mean(axis=0)
        keep_nan_mask = nan_fraction <= self.nan_threshold
        
        # Compute medians to fill remaining NaNs
        self.medians = np.zeros(D)
        for j in range(D):
            col = X[:, j]
            valid_vals = col[~np.isnan(col)]
            if len(valid_vals) > 0:
                self.medians[j] = np.median(valid_vals)
            else:
                self.medians[j] = 0.0
                
        # Fill NaNs temporarily for variance and correlation
        X_filled = X.copy()
        for j in range(D):
            nan_mask = np.isnan(X_filled[:, j])
            X_filled[nan_mask, j] = self.medians[j]
            
        # 2. Variance Threshold
        variances = np.var(X_filled, axis=0)
        keep_var_mask = (variances >= self.var_threshold) & keep_nan_mask
        
        indices_after_var = np.where(keep_var_mask)[0]
        
        if len(indices_after_var) == 0:
            raise ValueError("All descriptors were filtered out by NaN/Variance thresholds!")
            
        X_filtered = X_filled[:, indices_after_var]
        
        # 3. Correlation filter: Pearson correlation
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
    def __init__(self, feature_filter, scaler):
        self.feature_filter = feature_filter
        self.scaler = scaler
        
    def transform(self, X_desc: np.ndarray) -> np.ndarray:
        X_desc_filtered = self.feature_filter.transform(X_desc)
        X_desc_scaled = self.scaler.transform(X_desc_filtered)
        return X_desc_scaled


@lru_cache(maxsize=128)
def _morgan_bits(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)  # ExplicitBitVect
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.copy()


@lru_cache(maxsize=128)
def _maccs_bits(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = MACCSkeys.GenMACCSKeys(mol)
    n_bits = fp.GetNumBits()
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.copy()


CURATED_DESCRIPTORS_LIST = [
    # 1. Basic Physicochemical Properties
    "MolWt", "ExactMolWt", "HeavyAtomCount", "HeavyAtomMolWt",
    "MolLogP", "MolMR", "TPSA", "LabuteASA",

    # 2. Hydrogen Bonding & Charge polarities
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "FractionCSP3",
    "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge",

    # 3. Structural Rings & Heteroatoms
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
    "NumHeteroatoms", "NumValenceElectrons",

    # 4. Core topological descriptors (widely-validated in QSAR)
    "BalabanJ", "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0n", "Chi0v", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",

    # 5. EState indices — electrotopological state (key for GPCR binding interactions)
    "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
    "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11",
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex",

    # 6. VSA descriptors — surface area partitioned by logP/charge
    # (SlogP_VSA = Wildman-Crippen logP contribution per surface area bin)
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5",
    "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10",
    "SlogP_VSA11", "SlogP_VSA12",

    # 7. SMR_VSA — molar refractivity per surface area (polarizability profile)
    "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
    "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10",

    # 8. PEOE_VSA — partial equalization of orbital electronegativity per surface area
    # (charge distribution on molecular surface — critical for electrostatic receptor interactions)
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
    "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10",
    "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14",

    # 9. Additional validated QSAR descriptors
    "NHOHCount", "NOCount",  # Nitrogen/oxygen counts (GPCR-relevant heteroatoms)
    "NumRadicalElectrons",
    "qed",  # Quantitative Estimate of Drug-likeness
]


_DESC_FUNCS = {}
for name in CURATED_DESCRIPTORS_LIST:
    func = getattr(Descriptors, name, None)
    if func is None:
        func = getattr(Lipinski, name, None)
    if func is None and name == "qed":
        try:
            from rdkit.Chem.QED import qed as _qed_func
            func = _qed_func
        except ImportError:
            pass
    _DESC_FUNCS[name] = func


@lru_cache(maxsize=128)
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
            except Exception:
                logger.warning("Descriptor %s computation failed for SMILES %s, using NaN", name, smiles)
                vals.append(np.nan)
        else:
            logger.warning("Descriptor %s not found in RDKit build, using NaN", name)
            vals.append(np.nan)
    return np.array(vals, dtype=np.float32)



def build_feature_matrix(train_df, test_df, smiles_col: str = "canonical_smiles", save_to_disk: bool = True):
    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_df[smiles_col].tolist()
    from joblib import Parallel, delayed

    logger.info("Computing Morgan fingerprints (%d train, %d test)...", len(train_smiles), len(test_smiles))
    Xfp_train = np.vstack(Parallel(n_jobs=-1)(delayed(_morgan_bits)(s) for s in train_smiles))
    Xfp_test = np.vstack(Parallel(n_jobs=-1)(delayed(_morgan_bits)(s) for s in test_smiles))

    logger.info("Computing MACCS keys...")
    Xmaccs_train = np.vstack(Parallel(n_jobs=-1)(delayed(_maccs_bits)(s) for s in train_smiles))
    Xmaccs_test = np.vstack(Parallel(n_jobs=-1)(delayed(_maccs_bits)(s) for s in test_smiles))

    X_fp_train = np.hstack([Xfp_train, Xmaccs_train])
    X_fp_test = np.hstack([Xfp_test, Xmaccs_test])

    # Cache RDKit bitvectors for fast AD at inference
    def _get_fp(s):
        mol = Chem.MolFromSmiles(s)
        if mol:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            return generator.GetFingerprint(mol)
        return None
    train_fps = Parallel(n_jobs=-1)(delayed(_get_fp)(s) for s in train_smiles)
    
    if save_to_disk:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR / "train_fps.pkl", "wb") as f:
            pickle.dump(train_fps, f)

    logger.info("Computing curated RDKit descriptors (%d)...", len(CURATED_DESCRIPTORS_LIST))
    Xdesc_train = np.vstack(Parallel(n_jobs=-1)(delayed(_all_descriptors)(s) for s in train_smiles))
    Xdesc_test = np.vstack(Parallel(n_jobs=-1)(delayed(_all_descriptors)(s) for s in test_smiles))

    logger.info("Filtering descriptors (NaN=%.2f, Var=%.2f, Corr=%.2f)...",
                 FEATURE_NAN_THRESHOLD, FEATURE_VAR_THRESHOLD, FEATURE_CORR_THRESHOLD)
    desc_names = list(CURATED_DESCRIPTORS_LIST)
    feature_filter = FeatureFilter(
        nan_threshold=FEATURE_NAN_THRESHOLD,
        var_threshold=FEATURE_VAR_THRESHOLD,
        corr_threshold=FEATURE_CORR_THRESHOLD,
    )
    feature_filter.fit(Xdesc_train, feature_names=desc_names)

    Xdesc_train_filtered = feature_filter.transform(Xdesc_train)
    Xdesc_test_filtered = feature_filter.transform(Xdesc_test)

    logger.info("Started with %d descriptors -> kept %d after filtering.", Xdesc_train.shape[1], Xdesc_train_filtered.shape[1])

    scaler = StandardScaler()
    Xdesc_train_s = scaler.fit_transform(Xdesc_train_filtered)
    Xdesc_test_s = scaler.transform(Xdesc_test_filtered)

    pipeline = FeaturePipeline(feature_filter, scaler)

    X_train = np.hstack([X_fp_train, Xdesc_train_s]).astype(np.float32)
    X_test = np.hstack([X_fp_test, Xdesc_test_s]).astype(np.float32)

    if save_to_disk:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR / "train_smiles.pkl", "wb") as f:
            pickle.dump(train_smiles, f)
        with open(PROCESSED_DATA_DIR / "test_smiles.pkl", "wb") as f:
            pickle.dump(test_smiles, f)

        with open(PROCESSED_DATA_DIR / "features_train.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with open(PROCESSED_DATA_DIR / "features_test.pkl", "wb") as f:
            pickle.dump(X_test, f)

    return X_train, X_test, pipeline


def build_features(smiles: str, pipeline) -> np.ndarray:
    fp = _morgan_bits(smiles)
    maccs = _maccs_bits(smiles)
    desc = _all_descriptors(smiles).reshape(1, -1)
    desc_s = pipeline.transform(desc).ravel()
    return np.hstack([fp.astype(np.float32), maccs.astype(np.float32), desc_s.astype(np.float32)])
