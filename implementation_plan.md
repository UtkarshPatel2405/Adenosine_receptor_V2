# Scientific Architecture & Data Leakage Audit

This plan outlines the critical data leakages, biological fallacies, and architectural bottlenecks currently making the model unreliable. It also proposes rigorous solutions to elevate the platform to publication-grade precision.

## User Review Required
> [!IMPORTANT]
> The current pipeline contains a severe "Mutual Decoy" biological fallacy that poisons the training data with false negatives, and a train/test split misalignment that invalidates the GNN vs XGBoost comparison. Fixing these will fundamentally alter the model's reported performance metrics. Please review the proposed architectural shift to 3D.

## Open Questions
> [!WARNING]
> Upgrading to an SE(3)-equivariant GNN requires generating 3D conformers for all 30,000+ molecules. This is computationally expensive but mandatory for state-of-the-art precision. Should we implement RDKit-based 3D conformer generation (ETKDG) directly into the pipeline, or stick to advanced 2D architectures but fix the data leaks?

## Identified Critical Flaws

### 1. Data Split Misalignment (Invalid GNN vs XGBoost Comparison)
- **The Bug:** `gnn_model.py` generates its Bemis-Murcko scaffold split on a *per-subtype* basis, while `retrain_production.py` generates the split on the *global* dataset. Because the pool of unique scaffolds differs, the random assignment to Train/Test differs completely.
- **The Impact:** The GNN's Test Set contains molecules that are in the XGBoost's Train Set, and vice versa. `evaluator.py` simply reads the GNN's self-reported metrics from a JSON file, meaning the UI is comparing models evaluated on completely different molecular sets. 

### 2. The Mutual Decoy Fallacy (False Negatives)
- **The Bug:** In `data_loader.py`, if a molecule is highly active on A₁ but has *no data* for A₂A, the pipeline artificially injects a "decoy" row for A₂A with `pChEMBL = 4.0` (inactive). 
- **The Impact:** Missing data $\neq$ inactive. A highly potent A₁ ligand is often highly potent at A₂A. By forcing missing data to be "inactive", the dataset is flooded with false negatives, forcing the model to learn contradictory Structure-Activity Relationships (SAR). This destroys the reliability of the QSAR models.

### 3. Outdated Architectures for Affinity
- **The Bug:** The model relies on 2D fingerprints (XGBoost) and a basic 2D Graph Isomorphism Network (GINE).
- **The Impact:** 2D QSAR and simple message passing cannot resolve complex spatial binding pockets or true stereochemical activity cliffs. Modern publication-grade benchmarks demand 3D structural awareness.

### 4. Single-Threaded Bottleneck (System Hangs)
- **The Bug:** `build_feature_matrix()` in `features.py` calculates 50 complex RDKit physicochemical descriptors for ~48,000 molecules (including decoys) using a sequential `for` loop. 
- **The Impact:** This takes 15+ minutes, causing the evaluator to hang.

---

## Proposed Changes

### Data Pipeline Core
Summary: Eliminate false negatives and fix the global split leakage.

#### [MODIFY] src/data_loader.py
- **Remove Mutual Decoy Fallacy:** Remove the logic that artificially assigns `pChEMBL=4.0` to unverified subtypes. We will only use explicit negatives from ChEMBL/GPCRdb and the verified structural P2Y decoys.

#### [MODIFY] src/scaffold_split.py
- **Enforce Single Source of Truth:** Save the global Train/Test scaffold split to disk (`data/processed/global_split.json`) during `retrain_production.py` so that `gnn_model.py` is forced to load and respect the exact same split.

#### [MODIFY] src/features.py
- **Parallelize Feature Extraction:** Wrap `_all_descriptors` and `_morgan_bits` in `joblib.Parallel` to reduce feature extraction time from 15 minutes to under 2 minutes.

### Architecture Upgrade (To State-of-the-Art)
Summary: Transition the GNN to a 3D-aware architecture for maximum precision.

#### [MODIFY] src/gnn_model.py
- **Implement 3D Conformer Generation:** Update `_prepare_data` to generate RDKit 3D conformers (ETKDGv3) for the molecular graphs.
- **Upgrade to 3D Message Passing:** Replace the basic 2D GINE with a 3D-aware architecture (e.g., SchNet or a lightweight SE(3)-equivariant analog) that incorporates 3D atomic coordinates and distance geometries.

#### [MODIFY] src/evaluator.py
- **Fix GNN Metric Parsing:** Stop reading the GNN metrics from the isolated `_gnn_report.json`. Instead, load the GNN model directly into memory and evaluate it on the exact same `Xte` (Test Set) as the XGBoost models to guarantee a 1-to-1 comparison.

## Verification Plan

### Automated Tests
- Run `python -m src.retrain_production` to ensure the global split is generated and saved.
- Run `python -m src.gnn_model --epochs 5` to verify the GNN loads the global split.
- Ensure the intersection of SMILES in GNN Train and XGBoost Test is exactly 0.

### Manual Verification
- Check the distribution of pChEMBL values after removing the mutual decoys to ensure the false negative spike at 4.0 is gone.
- Verify `evaluator.py` execution time drops significantly due to `joblib.Parallel`.
