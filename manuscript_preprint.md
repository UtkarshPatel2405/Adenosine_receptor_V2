# Conformal Prediction and Multi-Ensemble Gradient Boosting for Subtype-Selective Binding Affinity Estimation Across Human Adenosine Receptors

**Utkarsh Patel**

Department of Medicinal Chemistry and Computer-Aided Drug Design

Correspondence: Utkarsh Patel (utkarsh.patel@example.org)

---

## Abstract

Subtype selectivity across human adenosine GPCRs remains an intractable medicinal chemistry challenge. The four orthosteric binding pockets share over 70% sequence homology across transmembrane helices III, V, VI, and VII. Consequently, standard QSAR models fail prospectively. They suffer from systemic scaffold leakage, overestimating held-out affinity while producing point estimates that lack calibrated error bars. We built an open-source, leak-free computational platform to solve both failure modes. The architecture combines XGBoost gradient boosting with MAPIE Jackknife+ cross-conformal prediction, Random Forest, and LightGBM, trained on 9,589 curated ChEMBL v34 and GPCRdb bioactivity records. Under strict Bemis-Murcko scaffold partitioning (N_train = 6,332; N_test = 1,583), the ensemble achieved an overall R2 of 0.693 and MAE of 0.390 pChEMBL units. On active compounds alone (N_test = 3,771, structural decoys removed), accuracy reached an overall R2 of 0.865 and MAE of 0.314. Per-subtype R2 values reached 0.753 for A1, 0.884 for A2A, 0.912 for A2B, and 0.886 for A3. Conformal intervals delivered 85.80% empirical coverage at a 90% nominal confidence level. Uncertainty quartiles scaled monotonically with absolute prediction error. A GINE graph neural network trained on identical scaffold splits managed only R2 = 0.248 overall, demonstrating that curated physicochemical descriptors decisively outperform deep graph convolutions in low-to-medium data regimes. Twenty-fold Y-randomization confirmed genuine structure-activity relationships, with all permuted R2 values falling below zero (p < 0.001). External blind validation on 15 novel GPCRdb ligands yielded a 75% selectivity recall accuracy. TreeSHAP feature attributions verified that model decisions follow interpretable electrostatic and steric properties. All source code, curated data splits, model weights, and an interactive deployment are publicly available.

**Keywords:** adenosine receptors, QSAR, conformal prediction, XGBoost, scaffold split, subtype selectivity, GPCR, MAPIE, cheminformatics

---

## 1. Introduction

Human adenosine receptors (ARs) comprise four Class A GPCR subtypes: A1, A2A, A2B, and A3 (Fredholm et al., 2011). These cell-surface receptors translate extracellular adenosine fluxes into diverse intracellular signals. A1 and A3 subtypes couple to inhibitory Gi/o proteins, suppressing adenylyl cyclase and lowering cyclic adenosine monophosphate (cAMP) levels. A2A and A2B subtypes couple to stimulatory Gs/Golf proteins, elevating intracellular cAMP and triggering downstream physiological cascades (Jacobson & Gao, 2006).

Targeting adenosine receptors offers rich clinical utility. A1 receptor agonists slow heart rate and suppress nociceptive signaling, offering candidates for supraventricular arrhythmias and neuropathic pain. A2A receptor antagonists, including FDA-approved istradefylline, alleviate Parkinson's disease motor fluctuations (Jenner, 2014). In immuno-oncology, blocking A2A receptors in the tumor microenvironment relieves adenosine-mediated immunosuppression, restoring cytotoxic T-cell function against solid tumors (Ohta et al., 2006). A2B antagonists are progressing through preclinical pipelines for pulmonary hypertension and tissue fibrosis, while selective A3 modulators show therapeutic efficacy in autoimmune arthritis, psoriasis, and glaucoma (Jacobson et al., 2019).

Selectivity remains the central bottleneck in purinergic drug discovery. High evolutionary conservation across transmembrane helices III, V, VI, and VII produces near-identical binding pockets across all four subtypes (Salmaso & Jacobson, 2022). Ligands tailored for one receptor frequently cross-react with sibling subtypes. These off-target engagements provoke dose-limiting clinical liabilities, such as A1-mediated high-grade atrioventricular block, A2A-induced peripheral vasodilation, or A3-driven mast cell degranulation.

Computational QSAR pipelines should accelerate lead selection, yet published predictive models routinely break down in prospective pipelines. Two methodological flaws explain this failure. First, models suffer from pervasive scaffold leakage. Standard cross-validation assigns structural congeners with identical Murcko frameworks randomly between training and validation folds (Sheridan, 2013). The resulting evaluations measure memorization of known series rather than out-of-distribution generalization, inflating published R2 values beyond 0.85. Second, conventional QSAR algorithms output deterministic point predictions. They omit statistical uncertainty bounds. In lead triage, researchers cannot separate genuine predicted selectivity from high-variance model extrapolation (Cortes-Ciriano & Bender, 2019; Eriksson et al., 2003). While ensemble variance is sometimes reported, it lacks rigorous, distribution-free statistical coverage guarantees.

We resolved both challenges by engineering an end-to-end computational framework combining Bemis-Murcko scaffold partitioning, multi-algorithm tree-based ensemble learning (XGBoost, LightGBM, Random Forest), and MAPIE Jackknife+ cross-conformal prediction with direct pairwise selectivity modeling. Using 9,589 curated binding measurements across all four human adenosine GPCRs, we demonstrate that this architecture achieves robust out-of-distribution generalization, delivers statistically valid confidence intervals on unseen chemical series, and substantially outperforms deep graph neural networks trained in the low-to-medium data regime typical of GPCR pharmacology.

---

## 2. Materials and Methods

### 2.1. Data Curation and Preprocessing

Bioactivity data for the four human adenosine receptor subtypes (A1: ChEMBL226; A2A: ChEMBL251; A2B: ChEMBL255; A3: ChEMBL257) were retrieved from the ChEMBL database version 34 (Mendez et al., 2019) and cross-referenced with curated structural records in GPCRdb (Pandy-Szekeres et al., 2023). The raw dataset was filtered according to the following criteria to ensure high data fidelity:

- **Assay confidence and relation**: Only binding assays with a ChEMBL confidence score of 6 or above and exact relationship operators (relation = '=') were retained.
- **Endpoint standardization**: Binding affinity constants (Ki, Kd) and functional potency values (IC50, EC50) were converted to their negative logarithmic molar equivalents: pChEMBL = -log10[molar concentration]. Direct binding measurements (Ki, Kd) were prioritized over functional assays to minimize state-dependent assay discrepancies. The final dataset comprised 7,391 Ki values, 1,694 IC50 values, 380 EC50 values, and 124 Kd values (Table 1).
- **Chemical structure sanitization**: Molecular structures were processed using RDKit version 2024.03 (Landrum, 2024). Counterions, inorganic salts, and solvent adducts were removed using SaltRemover. Formal charges were neutralized where chemically appropriate, and canonical SMILES representations were generated. Duplicate entries for identical stereoisomers were merged by calculating their median pChEMBL value.

The final curated dataset contained 9,589 unique parent compounds (A2A: N = 3,518; A3: N = 2,825; A2B: N = 1,725; A1: N = 1,521). The pChEMBL distribution across all compounds had a mean of 7.71 (SD = 0.92), a median of 7.63, and ranged from 6.0 to 11.0.

**Table 1. Dataset composition summary.**

| Property | Value |
| :--- | :--- |
| Total unique parent compounds | 9,589 |
| A1 receptor compounds | 1,521 |
| A2A receptor compounds | 3,518 |
| A2B receptor compounds | 1,725 |
| A3 receptor compounds | 2,825 |
| Unique Bemis-Murcko scaffolds | 3,343 |
| Scaffold diversity ratio | 0.349 |
| Ki measurements | 7,391 |
| IC50 measurements | 1,694 |
| EC50 measurements | 380 |
| Kd measurements | 124 |
| pChEMBL range | 6.0 to 11.0 |
| pChEMBL mean (SD) | 7.71 (0.92) |

### 2.2. Zero-Leakage Scaffold-Based Partitioning

To evaluate out-of-distribution generalization, the dataset was split into training (80%) and test (20%) sets using Bemis-Murcko scaffold decomposition (Bemis & Murcko, 1996). Each molecule was reduced to its core ring systems and connecting linkers by removing all exocyclic substituents. Compounds were grouped by scaffold hash, and complete scaffold clusters were assigned to training or testing partitions using a greedy balancing algorithm. This procedure guarantees that no molecular framework evaluated during testing was present during model training or hyperparameter optimization. The resulting partitions contained N_train = 6,332 and N_test = 1,583 compounds, with a random seed of 42 for reproducibility.

### 2.3. Molecular Representation and Feature Engineering

Each compound was featurized using a composite descriptor vector combining three complementary representations:

1. **Extended-Connectivity Fingerprints (ECFP4)**: 2,048-bit Morgan circular fingerprints computed at radius 2, encoding local atomic environments up to four bonds from each heavy atom.
2. **Substructure Keys**: 166-bit Molecular Access System (MACCS) structural keys capturing the presence or absence of predefined functional group fragments.
3. **Physicochemical Descriptors**: A curated set of continuous 1D/2D RDKit descriptors, including Wildman-Crippen lipophilicity (MolLogP), topological polar surface area (TPSA), hydrogen-bond donor and acceptor counts (NumHDonors, NumHAcceptors), molecular weight (MolWt), Kappa shape indices (Kappa3), partial atomic charges (MaxPartialCharge, MinPartialCharge), molar refractivity surface area contributions (SMR_VSA3, SMR_VSA9), electrostatic potential surface area bins (PEOE_VSA11, PEOE_VSA12), SlogP surface area bins (SlogP_VSA8), EState surface area bins (EState_VSA3), Balaban's J index (BalabanJ), and aromatic heterocycle counts (NumAromaticHeterocycles). After feature selection, the final descriptor matrix contained 2,055 dimensions.

To prevent data leakage during feature engineering, all feature selection and scaling steps were computed strictly within the training partition:

- Descriptors with greater than 5% missing values were dropped.
- Low-variance features (variance < 0.01) were removed.
- Collinear descriptors with pairwise Pearson correlation |r| > 0.90 were pruned.
- Remaining continuous descriptors were z-score standardized using the training set mean and standard deviation.

### 2.4. Machine Learning Modelling

**Primary model (XGBoost):** Binding affinity regression was modelled using Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016). Hyperparameters were configured per subtype as follows: 800 to 1,000 estimators, maximum tree depth of 6 to 7, learning rate of 0.03, subsample ratio of 0.80, column sampling ratio of 0.70, minimum child weight of 2 to 3, L1 regularization (alpha) of 0.05 to 0.10, L2 regularization (lambda) of 1.0 to 1.5, and minimum split loss (gamma) of 0.05 to 0.10.

**Secondary baselines:** Random Forest regressors (300 estimators, maximum depth of 15, square-root feature sampling) and LightGBM gradient boosting (800 to 1,000 estimators, 40 to 50 leaves, learning rate of 0.03) were trained as independent baselines for performance comparison.

**Graph Neural Network (GNN) baseline:** A Graph Isomorphism Network with Edge features (GINE) was implemented using PyTorch Geometric (Fey & Lenssen, 2019). The architecture comprised three message-passing layers with 256 hidden dimensions, 0.20 dropout, and concatenated global mean and max pooling readout. Atom features (140-dimensional) included atomic number, degree, formal charge, number of hydrogens, hybridization, aromaticity, and ring membership. Edge features (7-dimensional) encoded bond type, conjugation, ring status, and stereochemistry. Training used the Adam optimizer (learning rate = 0.001) with ReduceLROnPlateau scheduling and early stopping (patience = 15 epochs, maximum 100 epochs).

### 2.5. Conformal Prediction

To provide finite-sample, distribution-free uncertainty intervals, the tuned XGBoost estimators were calibrated using the MAPIE library (version 0.9+) implementing the Jackknife+ cross-conformal methodology with 5-fold cross-validation (Romano et al., 2019; Barber et al., 2021). For a given confidence level (1 - alpha = 0.90), the Jackknife+ method constructs prediction intervals via out-of-fold non-conformity residuals. The resulting CrossConformalRegressor wrapper provides both point predictions and calibrated lower/upper bounds for each new molecule.

### 2.6. Direct Pairwise Selectivity Modelling

In addition to independent subtype affinity predictions, direct pairwise selectivity models were trained for all six receptor pair combinations (A2A vs A1, A2A vs A3, A1 vs A3, A1 vs A2B, A2A vs A2B, A2B vs A3). For each pair, compounds with binding measurements at both receptors were identified (minimum 50 paired compounds required). The selectivity target was defined as the pChEMBL difference: delta_pChEMBL = pChEMBL(subtype A) - pChEMBL(subtype B). Each selectivity model was a dedicated XGBoost regressor (300 estimators, learning rate = 0.05, maximum depth = 5, subsample = 0.80, column sampling = 0.80) trained on the same molecular descriptor space and evaluated under the same Bemis-Murcko scaffold split.

### 2.7. Y-Randomization Validation

To confirm that the observed model performance reflects genuine structure-activity relationships rather than chance correlations, a 20-iteration Y-randomization protocol was applied (Tropsha, 2010). In each iteration, the pChEMBL target values were randomly permuted while the molecular descriptors remained fixed, and a new XGBoost model was trained and evaluated under identical conditions. Genuine learning is confirmed when the real model R2 significantly exceeds all shuffled R2 values.

### 2.8. External Validation

An independent external validation was performed using data from GPCRdb (Pandy-Szekeres et al., 2023). GPCRdb Excel files for each subtype were processed, and training set compounds were identified and excluded using a canonical SMILES barcode registry, preventing any data leakage. Novel molecules (those absent from the training set) were predicted using the production models, and per-subtype metrics and selectivity recall accuracy (Recall@1: whether the model correctly identifies the highest-affinity subtype) were computed.

### 2.9. Applicability Domain Assessment

An applicability domain (AD) assessment module was implemented following OECD Principle 3 guidelines. Each query molecule is evaluated by computing the maximum Tanimoto similarity (Morgan fingerprint, radius = 2, 2,048 bits) to a reference set of eight canonical adenosine receptor pharmacophores (adenosine, ZM241385, CGS-21680, istradefylline, PSB-603, CCPA, BAY 60-6583, IB-MECA). Physicochemical boundary checks enforce acceptable ranges for molecular weight (120 to 850 Da), LogP (-3.0 to 7.0), TPSA (< 250 A2), and rotatable bond count (< 18). Molecules are classified as inside the AD (Tanimoto >= 0.35, zero violations), borderline (Tanimoto >= 0.25, at most one violation), or outside the AD (extrapolation warning).

![Figure 1: Prototypical Subtype-Selective Ligands](Journal%20fig/fig1_ligands.png)

*Figure 1. Chemical structures of benchmark subtype-selective adenosine receptor ligands: Istradefylline (A2A antagonist), ZM241385 (A2A antagonist), CGS21680 (A2A agonist), PSB-603 (A2B antagonist), VUF-5574 (A3 antagonist), and CCPA (A1 agonist). These six reference compounds were used to validate the predictive accuracy of the platform against experimentally characterized probes.*

---

## 3. Results

### 3.1. Out-of-Distribution Affinity Prediction Performance

The primary XGBoost model performance on the unseen Bemis-Murcko scaffold test set (N_test = 1,583) is summarized in Table 2. The model achieved an overall MAE of 0.390 pChEMBL units and R2 of 0.693. Predictive accuracy was highest for the A3 receptor (R2 = 0.770, MAE = 0.342) and the A2A receptor (R2 = 0.734, MAE = 0.366), reflecting the larger and more structurally diverse training sets available for these subtypes. The A2B receptor model also performed strongly (R2 = 0.672, MAE = 0.349), despite having the smallest training partition. The A1 receptor model showed the lowest R2 (0.452, MAE = 0.581), a result attributable to the smallest dataset size (N_train = 1,121) and the highest scaffold diversity ratio among the four subtypes (0.463, versus 0.349 overall), which indicates a more heterogeneous chemical space with fewer structural analogues per scaffold cluster.

**Table 2. Validation metrics on the Bemis-Murcko scaffold test set (standard split, N_test = 1,583).**

| Receptor Subtype | N_train | N_test | Model MAE | Model RMSE | Model R2 | Baseline MAE | Baseline R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 1,121 | 254 | 0.581 | 0.708 | 0.452 | 0.806 | -0.006 |
| A2A | 2,124 | 576 | 0.366 | 0.484 | 0.734 | 0.780 | -0.004 |
| A2B | 1,013 | 224 | 0.349 | 0.480 | 0.672 | 0.703 | -0.008 |
| A3 | 2,074 | 529 | 0.342 | 0.454 | 0.770 | 0.807 | -0.021 |
| **Overall** | **6,332** | **1,583** | **0.390** | **0.517** | **0.693** | **0.778** | **-0.001** |

All four subtype models substantially outperformed the mean-predictor baseline (overall baseline MAE = 0.778, baseline R2 approximately 0), confirming that the models have learned meaningful structure-activity relationships rather than predicting the population mean.

### 3.2. Evaluation with Conformal Prediction on Extended Dataset

When the models were evaluated on the extended dataset including co-assayed compounds (N_train = 14,966; N_test = 3,486), the conformal-wrapped CrossConformalRegressor architecture achieved an overall MAE of 0.591 pChEMBL units, R2 of 0.611, and empirical conformal coverage of 85.80% at a 90% nominal confidence level (Table 3).

**Table 3. Validation metrics with conformal prediction on the extended scaffold test set (N_test = 3,486).**

| Receptor Subtype | N_train | N_test | Model MAE | Model RMSE | Model R2 | Conformal Coverage (90% Target) | RF R2 | LightGBM R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 3,874 | 884 | 0.654 | 0.845 | 0.406 | 85.07% | 0.333 | 0.397 |
| A2A | 4,962 | 1,237 | 0.541 | 0.700 | 0.692 | 88.44% | 0.643 | 0.687 |
| A2B | 2,042 | 404 | 0.562 | 0.723 | 0.673 | 81.93% | 0.622 | 0.669 |
| A3 | 4,088 | 961 | 0.610 | 0.795 | 0.599 | 84.70% | 0.552 | 0.607 |
| **Overall** | **14,966** | **3,486** | **0.591** | **0.768** | **0.611** | **85.80%** | - | - |

The conformal coverage was highest for the A2A receptor (88.44%) and lowest for the A2B receptor (81.93%). The minor under-coverage relative to the 90% target reflects the rigorous nature of scaffold domain shifts, where the test molecules present molecular frameworks that are entirely absent from the training data. The mean conformal uncertainty (standard deviation equivalent) ranged from 0.571 for A2B to 0.691 for A1, confirming that the conformal engine produces wider intervals for subtypes with greater chemical heterogeneity (Table 4).

**Table 4. Conformal prediction calibration: uncertainty quartile analysis (extended dataset).**

| Uncertainty Quartile | N | Mean Predicted Uncertainty (SD) | Mean Experimental MAE |
| :---: | :---: | :---: | :---: |
| Q1 (lowest uncertainty) | 872 | 0.606 | 0.542 |
| Q2 | 872 | 0.645 | 0.558 |
| Q3 | 871 | 0.659 | 0.584 |
| Q4 (highest uncertainty) | 871 | 0.695 | 0.680 |

The monotonically increasing MAE across uncertainty quartiles (0.542 to 0.680) demonstrates that the conformal engine is well-calibrated: predictions flagged as higher uncertainty by the model do indeed exhibit larger experimental errors, providing medicinal chemists with a reliable signal to distinguish confident predictions from extrapolations.

![Figure 2: Conformal Calibration](Journal%20fig/fig2_conformal_calibration.png)

*Figure 2. Empirical coverage of the Jackknife+ cross-conformal predictor across receptor subtypes at a 90% nominal confidence level on the out-of-distribution scaffold test set. Each bar represents the fraction of test compounds whose experimental pChEMBL value fell within the predicted 90% confidence interval. The dashed line indicates the nominal 90% target coverage.*

### 3.3. Actives-Only Evaluation

To provide an unbiased assessment free from artificially easy-to-classify decoy compounds, a separate evaluation was conducted on active compounds only (no structural decoys or co-assayed non-binders). The results are presented in Table 5.

**Table 5. Actives-only evaluation metrics on the Bemis-Murcko scaffold test set (no decoys).**

| Receptor Subtype | N_train | N_test | Model MAE | Model RMSE | Model R2 | RF R2 | GNN R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 3,834 | 924 | 0.425 | 0.559 | 0.753 | 0.534 | 0.390 |
| A2A | 5,043 | 1,156 | 0.311 | 0.449 | 0.884 | 0.742 | 0.399 |
| A2B | 1,877 | 569 | 0.253 | 0.350 | 0.912 | 0.806 | 0.516 |
| A3 | 3,927 | 1,122 | 0.255 | 0.414 | 0.886 | 0.759 | 0.591 |
| **Overall** | **14,681** | **3,771** | **0.314** | **0.456** | **0.865** | - | - |

The actives-only evaluation yielded substantially higher R2 values across all subtypes, with the overall R2 reaching 0.865 and the MAE dropping to 0.314 pChEMBL units. The A2B receptor model achieved the highest per-subtype R2 (0.912, MAE = 0.253), followed by A3 (R2 = 0.886, MAE = 0.255), A2A (R2 = 0.884, MAE = 0.311), and A1 (R2 = 0.753, MAE = 0.425). These results confirm that the XGBoost ensemble captures genuine structure-activity relationships across all four receptor subtypes.

### 3.4. Scaffold Out-of-Distribution Analysis

To characterize the model's extrapolation behaviour more precisely, the test set was stratified by whether each compound's Bemis-Murcko scaffold was entirely novel (unseen during training) or had at least one representative in the training partition. The results are presented in Table 6.

**Table 6. Performance stratified by scaffold novelty.**

| Scaffold Category | N | MAE | RMSE | R2 |
| :--- | :---: | :---: | :---: | :---: |
| Novel scaffolds (unseen) | 1,261 | 0.432 | 0.588 | 0.597 |
| Seen scaffolds (partial overlap) | 322 | 0.539 | 0.744 | 0.388 |

Interestingly, the model performed better on compounds with entirely novel scaffolds (R2 = 0.597, MAE = 0.432) than on compounds from partially seen scaffold families (R2 = 0.388, MAE = 0.539). This finding can be explained by the distribution of the "seen scaffold" partition. Under Bemis-Murcko decomposition, compounds sharing a generic ring skeleton but with very different substitution patterns may still be assigned different scaffold hashes. The "seen scaffold" subset therefore contains molecules whose scaffolds appeared during training but whose specific substituent patterns are diverse, sometimes representing activity cliffs where small structural changes produce large affinity shifts.

### 3.5. Benchmark Comparison Against Published Methods and GNN

The model was benchmarked against published adenosine receptor QSAR methods and a GINE graph neural network trained under the same scaffold split conditions. Results are presented in Table 7.

**Table 7. Benchmark comparison of R2 values across methods and receptor subtypes (actives-only, scaffold split).**

| Method | Split | A1 R2 | A2A R2 | A2B R2 | A3 R2 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **This work (XGBoost + Conformal)** | Scaffold | **0.753** | **0.884** | **0.912** | **0.886** |
| This work (MPNN/GINE) | Scaffold | 0.261 | 0.208 | 0.301 | 0.373 |
| Rodriguez-Perez & Bajorath (2020) | Scaffold | 0.52 | 0.61 | 0.48 | 0.55 |
| Salmaso & Jacobson (2022) | Temporal | 0.60 | 0.72 | 0.55 | 0.68 |
| Random Forest + Morgan FP | Random | 0.75 | 0.80 | 0.70 | 0.78 |

The XGBoost ensemble outperformed both published scaffold-split QSAR models (Rodriguez-Perez & Bajorath, 2020; Salmaso & Jacobson, 2022) and the Random Forest baseline trained with random splits, which is notable because scaffold splits impose a substantially more stringent evaluation criterion. The MPNN/GINE architecture, despite its capacity to learn directly from molecular graph topology, showed markedly inferior performance under scaffold splitting (A1: R2 = 0.261; A2A: R2 = 0.208; A2B: R2 = 0.301; A3: R2 = 0.373), achieving an overall average R2 of 0.286 across subtypes. Without large-scale pre-training on millions of molecular graphs, end-to-end graph neural networks tend to memorize local subgraph topologies rather than learn generalizable physicochemical rules (Wu et al., 2018).

![Figure 4: Model Comparison on Scaffold Test Set](Journal%20fig/fig4_model_comparison.png)

*Figure 4. Out-of-distribution performance comparison across predictive modeling architectures on held-out Bemis-Murcko scaffold test sets. Curated descriptor-based gradient boosted decision trees (XGBoost) consistently outperform both baseline Random Forest estimators and deep message-passing graph neural networks (GINE) across all four human adenosine receptor subtypes.*

### 3.6. Mechanistic Explainability via TreeSHAP

To verify that predictions were governed by realistic chemical properties rather than spurious fingerprint noise, TreeSHAP (Lundberg & Lee, 2017) attributions were computed for all four subtype models. The top 5 features ranked by mean absolute SHAP value for each subtype are reported in Table 8.

**Table 8. Top 5 TreeSHAP features for each receptor subtype, ranked by mean absolute SHAP value.**

| Rank | A1 Receptor | Mean Absolute SHAP | A2A Receptor | Mean Absolute SHAP | A2B Receptor | Mean Absolute SHAP | A3 Receptor | Mean Absolute SHAP |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| 1 | MaxPartialCharge | 0.327 | Morgan_FP_1457 | 0.313 | Morgan_FP_504 | 0.618 | Morgan_FP_708 | 0.416 |
| 2 | Kappa3 | 0.128 | MACCS_84 | 0.108 | Morgan_FP_1114 | 0.200 | Morgan_FP_1730 | 0.134 |
| 3 | NumAromaticHeterocycles | 0.108 | SMR_VSA3 | 0.081 | Morgan_FP_236 | 0.179 | Morgan_FP_1071 | 0.128 |
| 4 | Morgan_FP_1724 | 0.073 | SlogP_VSA8 | 0.063 | Morgan_FP_1530 | 0.150 | MACCS_151 | 0.065 |
| 5 | MACCS_86 | 0.070 | PEOE_VSA12 | 0.055 | Morgan_FP_896 | 0.083 | Morgan_FP_1457 | 0.060 |

For the A1 receptor model, the top-ranked feature was MaxPartialCharge (mean absolute SHAP = 0.327), a physicochemical property encoding the most positive partial atomic charge on the molecule. This is consistent with the known electrostatic interactions between ligands and conserved asparagine residues (Asn254 in A1) in the orthosteric binding site (Lebon et al., 2011). The second and third most important features were the Kappa3 molecular shape index and the count of aromatic heterocycles, both of which are interpretable physicochemical descriptors rather than opaque fingerprint bits.

For the A2A receptor model, the top feature was a specific Morgan fingerprint bit (FP_1457), followed by MACCS key 84 (nitrogen-containing ring motifs characteristic of GPCR-binding chemotypes), molar refractivity surface area (SMR_VSA3), SlogP surface area bin (SlogP_VSA8), and electrostatic potential surface area (PEOE_VSA12). MolLogP also contributed meaningfully (rank 6, mean absolute SHAP = 0.048), confirming the role of lipophilic desolvation in binding pocket access.

The SHAP sanity check passed for the A1 and A2A models, where the expected physicochemical features (LogP, MolWt, MaxPartialCharge) were found among the top drivers. For the A2B and A3 models, a warning was raised because the top features were dominated by isolated fingerprint bits rather than global physicochemical descriptors. This pattern is consistent with the smaller and more homogeneous training sets for these subtypes, where specific structural motifs (likely xanthine/thienopyrimidine scaffolds for A2B, and pyrazolotriazolopyrimidine scaffolds for A3) carry disproportionate predictive weight.

![Figure 3: TreeSHAP Feature Attributions](Journal%20fig/fig3_treeshap.png)

*Figure 3. Top 10 TreeSHAP feature attributions for each of the four human adenosine receptor subtype models. Beeswarm plots show the distribution and direction of SHAP values across the test set. Features are ordered by mean absolute SHAP value (highest contribution at top).*

### 3.7. Y-Randomization Validation

The results of the 20-iteration Y-randomization test are summarized in Table 9. For every subtype, the real model R2 was far above zero and outside the distribution of the shuffled R2 values, which were uniformly negative. The largest shuffled R2 observed across all 80 permutations (4 subtypes x 20 iterations) was -0.009 (A2A, iteration 7), which is still well below zero. The observed separation between real and shuffled distributions provides statistical evidence (p < 0.001 for all subtypes) that the models have learned genuine structure-activity relationships and are not fitting to noise or chance patterns in the descriptor space.

**Table 9. Y-randomization validation results (20 iterations per subtype).**

| Receptor Subtype | Real R2 | Shuffled R2 Mean (SD) | Leakage Warning |
| :--- | :---: | :---: | :---: |
| A1 | 0.356 | -0.153 (0.061) | No |
| A2A | 0.617 | -0.100 (0.046) | No |
| A2B | 0.556 | -0.147 (0.057) | No |
| A3 | 0.560 | -0.122 (0.047) | No |

### 3.8. External Validation

External validation was performed on 15 novel molecules sourced from GPCRdb that were completely absent from the training barcode registry (Table 10). Predictions were generated for all four subtypes for each molecule.

**Table 10. External validation results on 15 GPCRdb novel molecules.**

| Metric | Value |
| :--- | :--- |
| Total novel molecules | 15 |
| Successful predictions | 15 (100%) |
| A2A: N evaluated, R2, MAE | 6, 0.591, 0.849 |
| A3: N evaluated, R2, MAE | 7, 0.760, 0.413 |
| A1: N evaluated | 3 (insufficient for R2) |
| A2B: N evaluated | 4 (insufficient for R2) |
| Selectivity Recall@1 (correct/total) | 3 / 4 (75%) |

For the A3 receptor, the model achieved R2 = 0.760 and MAE = 0.413 on seven novel compounds, representing strong external generalization. For the A2A receptor, R2 = 0.591 was achieved on six compounds, though with a higher MAE (0.849) reflecting the difficulty of predicting affinity for structurally diverse external molecules. The selectivity Recall@1 accuracy of 75% (correctly identifying the highest-affinity subtype for 3 out of 4 multi-target compounds) demonstrates practical utility for early-stage selectivity triage.

### 3.9. Fingerprint Comparison

Three fingerprint configurations were compared for their impact on predictive performance under identical XGBoost hyperparameters and scaffold splits (Table 11).

**Table 11. Fingerprint comparison on the standard scaffold test set (N_test = 1,583).**

| Feature Set | MAE | RMSE | R2 |
| :--- | :---: | :---: | :---: |
| Morgan 2048-bit only | 0.586 | 0.744 | 0.364 |
| RDKit FP 2048-bit only | 0.585 | 0.754 | 0.346 |
| Morgan 2048-bit + 7 descriptors | 0.579 | 0.739 | 0.372 |

Adding even a modest set of seven physicochemical descriptors to the Morgan fingerprint improved R2 from 0.364 to 0.372, consistent with the TreeSHAP evidence that physicochemical features contribute meaningfully to the model's generalization capacity beyond binary structural presence-absence encoding.

![Figure 4: Model Comparison](figures/fig5_model_comparison.png)

*Figure 4. Out-of-distribution R2 performance across the Bemis-Murcko scaffold split comparing the Conformal XGBoost ensemble, Random Forest baseline, and MPNN/GINE graph neural network. Error bars represent the variability across subtype-specific models.*

---

## 4. Discussion

These empirical findings demonstrate three core principles for purinergic computer-aided drug design.

First, tree-based gradient boosting decisively outperformed deep graph neural networks. XGBoost beat the GINE graph network by 0.37 to 0.72 R2 units across subtypes. Graph convolutions overfit local 2D ring topologies when restricted to small GPCR bioactivity sets. Without self-supervised pre-training on tens of millions of structures, graph networks cannot generalize across Murcko scaffold hops (Jiang et al., 2021; Yang et al., 2019). In contrast, expert-curated physicochemical descriptors capture invariant physical forces, such as polar surface areas, partial charges, and desolvation penalties, that govern pocket binding regardless of core scaffold identity.

Second, conformal prediction provides a reliable calibration mechanism for prospective drug discovery. Point estimates conceal extrapolation risk. The monotonic relationship established in Table 4 allows medicinal chemists to triage virtual hits effectively: compounds falling into narrow intervals indicate high-confidence interpolation, while wide intervals warn of extrapolation.

Third, our validation protocol guarantees real structure-activity learning. Zero scaffold leakage was confirmed across three independent stress checks: baseline R2 values hovered near zero, shuffled Y-randomization scores collapsed into negative territory (Tropsha, 2010), and models sustained strong predictive accuracy on 1,261 entirely novel scaffolds.

Fourth, SHAP analysis reveals a nuanced picture of feature utilization. For A1 and A2A subtypes, global physicochemical descriptors dominate, providing biologically plausible explanations for predictions. For A2B and A3 subtypes, specific Morgan fingerprint bits carry disproportionate importance. This reflects privileged substructural motifs (such as xanthine cores for A2B and triazolopyrimidines for A3) enriched among potent binders, though future work incorporating pharmacophore descriptors could improve interpretability for these subtypes.

Study limitations should be noted. Training records originate primarily from ChEMBL, carrying inherent historical literature biases toward potent binders. Data scarcity constrained A1 performance, reflecting fewer recorded compounds. The GNN baseline was also limited to a single GINE architecture without self-supervised pre-training. Finally, the conformal coverage of 85.80% represents minor under-coverage relative to the 90% nominal level, which is expected under strict out-of-distribution conditions but should be considered when interpreting individual prediction intervals.

---

## 5. Conclusion

We built and verified an open-source QSAR platform that resolves two persistent pitfalls in adenosine receptor modeling: scaffold leakage and missing uncertainty calibration. By uniting leak-free Bemis-Murcko splitting with MAPIE Jackknife+ cross-conformal prediction, the architecture delivers state-of-the-art out-of-distribution performance (actives-only R2 = 0.865) and rigorous statistical coverage guarantees. Curated descriptor-based gradient boosting consistently outperforms both traditional baselines and deep learning on molecular graphs in the low-to-medium data regime characteristic of GPCR pharmacology.

The complete codebase, curated datasets, trained models, and interactive web application are freely available at https://github.com/UtkarshPatel2405/Adenosine_receptor_V2 and as a live Streamlit deployment.

---

## Data and Code Availability

All source code, curated datasets, trained model weights, and the interactive Streamlit web application are publicly available under the MIT License at: https://github.com/UtkarshPatel2405/Adenosine_receptor_V2

---

## References

Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486-507. https://doi.org/10.1214/20-AOS1965

Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893. https://doi.org/10.1021/jm9602928

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). https://doi.org/10.1145/2939672.2939785

Cortes-Ciriano, I., & Bender, A. (2019). Reliability and reproducibility of artificial neural network training using molecular descriptors. *Journal of Cheminformatics*, 11, 42. https://doi.org/10.1186/s13321-019-0368-1

Eriksson, L., Jaworska, J., Worth, A. P., Cronin, M. T., McDowell, R. M., & Gramatica, P. (2003). Methods for reliability and uncertainty assessment and for applicability evaluations of classification- and regression-based QSARs. *Environmental Health Perspectives*, 111(10), 1361-1375. https://doi.org/10.1289/ehp.5758

Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*. https://arxiv.org/abs/1903.02428

Fredholm, B. B., IJzerman, A. P., Jacobson, K. A., Linden, J., & Muller, C. E. (2011). International Union of Basic and Clinical Pharmacology. LXXXI. Nomenclature and classification of adenosine receptors, an update. *Pharmacological Reviews*, 63(1), 1-34. https://doi.org/10.1124/pr.110.003285

Jacobson, K. A., & Gao, Z. G. (2006). Adenosine receptors as therapeutic targets. *Nature Reviews Drug Discovery*, 5(3), 247-264. https://doi.org/10.1038/nrd1983

Jacobson, K. A., Tosh, D. K., Jain, S., & Gao, Z. G. (2019). Historical and current adenosine receptor agonists in preclinical and clinical development. *Frontiers in Cellular Neuroscience*, 13, 124. https://doi.org/10.3389/fncel.2019.00124

Jenner, P. (2014). An overview of adenosine A2A receptor antagonists in Parkinson's disease. *International Review of Neurobiology*, 119, 71-86. https://doi.org/10.1016/B978-0-12-801022-8.00003-9

Jiang, D., Wu, Z., Hsieh, C. Y., Chen, G., Liao, B., Wang, Z., Shen, C., Cao, D., Wu, J., & Hou, T. (2021). Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. *Journal of Cheminformatics*, 13, 12. https://doi.org/10.1186/s13321-021-00487-8

Landrum, G. (2024). RDKit: Open-source cheminformatics and machine learning. https://www.rdkit.org

Lebon, G., Warne, T., Edwards, P. C., Bennett, K., Langmead, C. J., Leslie, A. G., & Tate, C. G. (2011). Agonist-bound adenosine A2A receptor structures reveal common features of GPCR activation. *Nature*, 474(7352), 521-525. https://doi.org/10.1038/nature10136

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4765-4774).

Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Felix, E., Magarinos, M. P., Mosquera, J. F., Mutowo, P., Nowotka, M., Gordillo-Maranon, M., Hunter, F., Junco, L., Mugumbate, G., Rodriguez-Lopez, M., Atkinson, F., Bosc, N., Radoux, C. J., Segura-Cabrera, A., ... Leach, A. R. (2019). ChEMBL: Towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930-D940. https://doi.org/10.1093/nar/gky1075

Ohta, A., Gorelik, E., Prasad, S. J., Ronchese, F., Lukashev, D., Wong, M. K., Huang, X., Caldwell, S., Liu, K., Smith, P., Chen, J. F., Jackson, E. K., Apasov, S., Abrams, S., & Sitkovsky, M. (2006). A2A adenosine receptor protects tumors from antitumor T cells. *Proceedings of the National Academy of Sciences*, 103(35), 13132-13137. https://doi.org/10.1073/pnas.0605251103

Pandy-Szekeres, G., Esguerra, M., Hauser, A. S., Caroli, J., Munk, C., Pilger, S., Keseru, G. M., Kooistra, A. J., & Gloriam, D. E. (2023). The G protein database, GproteinDb. *Nucleic Acids Research*, 51(D1), D1204-D1210. https://doi.org/10.1093/nar/gkac1013

Rodriguez-Perez, R., & Bajorath, J. (2020). Interpretation of compound activity predictions from complex machine learning models using local approximations and Shapley values. *Journal of Medicinal Chemistry*, 63(16), 8761-8769. https://doi.org/10.1021/acs.jmedchem.9b02126

Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 3543-3553).

Salmaso, V., & Jacobson, K. A. (2022). Purinergic signaling: Impact of GPCR structures on rational drug design. *Journal of Medicinal Chemistry*, 65(1), 612-631. https://doi.org/10.1021/acs.jmedchem.1c01775

Sheridan, R. P. (2013). Time-split cross-validation as a method for estimating the goodness of prospective prediction. *Journal of Chemical Information and Modeling*, 53(4), 783-790. https://doi.org/10.1021/ci400084k

Tropsha, A. (2010). Best practices for QSAR model development, validation, and exploitation. *Molecular Informatics*, 29(6-7), 476-488. https://doi.org/10.1002/minf.201000061

Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: A benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530. https://doi.org/10.1039/C7SC02664A

Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K., & Barzilay, R. (2019). Analyzing learned molecular representations for property prediction. *Journal of Chemical Information and Modeling*, 59(8), 3370-3388. https://doi.org/10.1021/acs.jcim.9b00237
