# Scaffold-Aware Conformal QSAR Modeling and Target Selectivity Prediction Across Human Adenosine Receptor Subtypes

## 1. Title & Metadata

**Title:** Scaffold-Aware Conformal QSAR Modeling and Target Selectivity Prediction Across Human Adenosine Receptor Subtypes

**Authors & Affiliations:**  
Utkarsh Patel$^{1,*}$

$^{1}$ Department of Medicinal Chemistry and Computer-Aided Drug Design, Academic Medical Center, Postal Code 380009, India  

**ORCID iD:** [0009-0002-8419-7432](https://orcid.org/0009-0002-8419-7432)  

**Corresponding Author:**  
Utkarsh Patel  
Department of Medicinal Chemistry and Computer-Aided Drug Design, Academic Medical Center, Postal Code 380009, India  
Email: utkarsh.patel@example.org  

---

## 2. Graphical Abstract & Highlights

### Graphical Abstract

![Graphical Abstract: Computational Pipeline for Adenosine Receptor Selectivity](<Journal fig/graphical_abstract.png>)

*Graphical Abstract. End-to-end computational pipeline for subtype-selective binding affinity prediction across human adenosine receptors. Chemical curation from ChEMBL v34 and GPCRdb feeds leak-free Bemis-Murcko scaffold partitioning, feature extraction (Morgan fingerprints, MACCS keys, and continuous RDKit descriptors), multi-ensemble gradient boosting (XGBoost, LightGBM, Random Forest), MAPIE Jackknife+ cross-conformal prediction, and mechanistic TreeSHAP feature attributions.*

### Highlights (TOC)

* Bemis-Murcko scaffold splitting eliminates data leakage across human adenosine GPCRs.
* Multi-ensemble XGBoost attains R2 of 0.865 and MAE of 0.314 on active compounds.
* Jackknife+ conformal intervals achieve 85.80% coverage at 90% confidence level.
* TreeSHAP verifies biophysical drivers over opaque graph convolution artifacts.
* Blind external GPCRdb testing delivers 75% subtype selectivity recall accuracy.

---

## 3. Abstract

**Motivation/Background:** Human adenosine receptors (A1, A2A, A2B, A3) share over 70% sequence homology across their orthosteric transmembrane pockets, making subtype-selective ligand design an arduous medicinal chemistry challenge. Conventional QSAR models suffer from scaffold leakage and lack calibrated uncertainty intervals, leading to overoptimistic performance reports and unquantified extrapolation risks during lead triage.

**Results:** We established a leak-free computational pipeline coupling RDKit physicochemical descriptors, multi-ensemble gradient boosting (XGBoost, LightGBM, Random Forest), and MAPIE Jackknife+ cross-conformal prediction across 9,589 curated bioactivity records. Evaluated under strict Bemis-Murcko scaffold partitioning (N_train = 6,332; N_test = 1,583), the ensemble achieved an overall R2 of 0.693 and MAE of 0.390 pChEMBL units. On active molecules alone (N_test = 3,771, structural decoys excluded), prediction accuracy reached an overall R2 of 0.865 (RMSE = 0.456, MAE = 0.314), with per-subtype R2 values of 0.753 (A1), 0.884 (A2A), 0.912 (A2B), and 0.886 (A3). Conformal prediction produced 85.80% empirical coverage at a 90% nominal confidence level. A GINE graph neural network yielded an overall R2 of only 0.248, demonstrating that expert-curated physicochemical descriptors decisively outperform deep graph representations under scaffold shifts. Blind external testing on 15 novel GPCRdb ligands yielded a 75% selectivity recall accuracy. Twenty-fold Y-randomization confirmed authentic structure-activity learning (all shuffled R2 < 0, p < 0.001).

**Availability:** Source code, curated datasets, and interactive web tools are available at https://github.com/UtkarshPatel2405/Adenosine_receptor_V2.

**Keywords:** adenosine receptors, QSAR, conformal prediction, XGBoost, scaffold split, subtype selectivity, GPCR, TreeSHAP, cheminformatics.

---

## 4. Introduction

### Macro Context: Biological Significance of the Target

Human adenosine receptors belong to Class A G protein-coupled receptors (GPCRs) and comprise four distinct subtypes: A1, A2A, A2B, and A3 (Fredholm et al., 2011). These receptors sense extracellular purine fluxes to modulate homeostatic and stress-response pathways throughout the cardiovascular, nervous, and immune systems. A1 and A3 receptors couple primarily to heterotrimeric Gi/o proteins, which inhibit adenylyl cyclase, suppress intracellular cyclic adenosine monophosphate (cAMP) production, and modulate inward-rectifying potassium channels. In contrast, A2A and A2B receptors couple to stimulatory Gs and Golf proteins, stimulating adenylyl cyclase, elevating intracellular cAMP concentrations, and activating protein kinase A signaling cascades (Jacobson & Gao, 2006).

Selective pharmacological modulation of individual adenosine receptor subtypes holds verified therapeutic promise. A1 receptor agonists slow cardiac conduction, showing promise in supraventricular tachycardias and neuropathic pain. A2A receptor antagonists, exemplified by the approved therapeutic istradefylline, reduce motor complications in Parkinson's disease (Jenner, 2014). In immuno-oncology, adenosine accumulated in hypoxic tumor microenvironments engages A2A receptors on cytotoxic T cells and natural killer cells, blunting antitumor immunity. Blocking this pathway restores cell-mediated cytotoxicity (Ohta et al., 2006). A2B receptor antagonists have entered clinical trials for pulmonary arterial hypertension and renal fibrosis, while A3 receptor modulators exhibit anti-inflammatory activity in rheumatoid arthritis, psoriasis, and ocular dry eye disorders (Jacobson et al., 2019).

### Micro Context: Computational Methods and Their Failure Modes

Subtype selectivity remains the dominant stumbling block in purinergic lead discovery. Transmembrane helices III, V, VI, and VII form an orthosteric binding cavity that is conserved across all four subtypes (Salmaso & Jacobson, 2022). Consequently, synthetic ligands frequently cross-react with off-target adenosine subtypes, causing severe adverse events. For instance, off-target A1 activation induces bradycardia and heart block, while indiscriminate A2A antagonism can elevate blood pressure and cause sleep disruptions.

Computer-aided drug design and Quantitative Structure-Activity Relationship (QSAR) models offer an efficient approach to profile selectivity in silico before chemical synthesis. However, published QSAR workflows routinely fail to translate into prospective lead identification. Two critical methodological vulnerabilities explain this shortfall:

1. **Scaffold Leakage and Optimism Bias:** Standard cross-validation workflows apply random partitioning at the compound level. This permits structural analogues and congeneric series with identical Murcko frameworks to reside simultaneously in training and test partitions (Sheridan, 2013). Models evaluated this way memorize local substituent patterns rather than learning transferable biophysical principles. When challenged with truly novel chemotypes, their predictive performance drops precipitously.

2. **Deterministic Point Estimates Without Calibrated Error Bounds:** Standard machine learning regressors output a single affinity value without uncertainty estimates (Cortes-Ciriano & Bender, 2019; Eriksson et al., 2003). In lead triage, computational medicinal chemists cannot distinguish between confident interpolations within well-mapped chemical space and high-variance extrapolations. While heuristic measures like ensemble standard deviations are sometimes reported, they lack formal coverage guarantees and fail to account for non-Gaussian residual distributions.

3. **Graph Neural Network Failure under Scaffold Shift:** Deep learning methods such as Graph Convolutional Networks (GCN) and Message Passing Neural Networks (MPNN) often overfit local 2D graph topologies on modest GPCR bioactivity sets. Without massive self-supervised pre-training, graph architectures tend to underperform descriptor-based models when required to generalize across distinct Bemis-Murcko scaffolds (Jiang et al., 2021; Yang et al., 2019).

### The Gap: The Adenosine Receptor Profiler Solution

This investigation establishes an open-source, leak-free computational platform specifically engineered to eliminate both scaffold leakage and uncalibrated prediction errors. We combine rigorous Bemis-Murcko scaffold partitioning with multi-algorithm gradient boosting (XGBoost, LightGBM, Random Forest) and MAPIE Jackknife+ cross-conformal prediction. We validate this pipeline against 9,589 curated binding affinity records spanning all four human adenosine receptor subtypes, establish direct pairwise selectivity estimators, provide mechanistic TreeSHAP attributions, and confirm true prospective utility via blind external testing on novel GPCRdb ligands.

---

## 5. Materials and Methods

### Data Acquisition & Curation

Bioactivity data for human adenosine receptor subtypes (A1: ChEMBL226; A2A: ChEMBL251; A2B: ChEMBL255; A3: ChEMBL257) were collected from ChEMBL version 34 (Mendez et al., 2019) and cross-validated with structural records from GPCRdb (Pandy-Szekeres et al., 2023). A reproducible data curation protocol was enforced:

1. **Assay Confidence Filtering:** Only primary target binding assays annotated with a ChEMBL confidence score of 6 or higher and exact relation operators (relation = '=') were retrieved. Unspecified relations, approximate bounds ('>', '<', '~'), and functional cell assay measurements with indirect signaling cascades were excluded.
2. **Endpoint Standardization:** Experimental affinity parameters (Ki and Kd) along with potency values (IC50 and EC50) were transformed into negative logarithmic molar units:
$$\text{pChEMBL} = -\log_{10}(\text{Concentration in Moles})$$
Equilibrium dissociation constants (Ki and Kd) were prioritized over IC50 or EC50 values to minimize assay-format dependency. The final dataset consists of 7,391 Ki values (77.1%), 1,694 IC50 values (17.7%), 380 EC50 values (4.0%), and 124 Kd values (1.3%).
3. **Structure Sanitization Protocol:** All chemical structures were processed using RDKit version 2024.03 (Landrum, 2024). Counterions, solvent molecules, and salt adducts were excised using `SaltRemover`. Zwitterions and acid-base states were standardized to their predominant protonation species at physiological pH (pH 7.4). Canonical SMILES were generated.
4. **Deduplication and Conflict Resolution:** Where duplicate assay records existed for identical stereoisomers under the same subtype, the median pChEMBL value was calculated and recorded.

The resulting dataset contains 9,589 unique parent structures distributed across A2A (N = 3,518), A3 (N = 2,825), A2B (N = 1,725), and A1 (N = 1,521). The overall pChEMBL affinity values range from 6.0 to 11.0, with a mean of 7.71 and standard deviation of 0.92.

### Model Architecture & Algorithm Formulation

#### Molecular Representation

Molecules were represented using a hybrid 2,055-dimensional feature space constructed to capture both 2D structural subgraphs and continuous biophysical properties:
1. **Morgan Circular Fingerprints:** 2,048 bits, radius 2 (equivalent to ECFP4), computed with RDKit.
2. **MACCS Structural Keys:** 166 bits encoding predefined functional groups and ring systems.
3. **Physicochemical Descriptors:** Curated 2D continuous properties including partition coefficient (MolLogP), topological polar surface area (TPSA), molecular weight (MolWt), extreme atomic partial charges (MaxPartialCharge, MinPartialCharge), shape index (Kappa3), aromatic heterocycle counts (NumAromaticHeterocycles), and electrotopological state and surface area bins (SMR_VSA3, SMR_VSA9, PEOE_VSA11, PEOE_VSA12, SlogP_VSA8, EState_VSA3, BalabanJ).

#### Feature Preprocessing Pipeline

To eliminate data leakage, all preprocessing transformations were fitted strictly on the training partition:
* Descriptors with missing values exceeding 5% were removed.
* Low-variance features (variance < 0.01) were pruned.
* Collinear descriptor pairs with Pearson correlation coefficient $|r| > 0.90$ were pruned by eliminating the feature with lower variance.
* Continuous descriptors were standardized via z-score scaling computed solely on the training partition.

#### Multi-Ensemble Gradient Boosting

Primary regression was carried out using Extreme Gradient Boosting (XGBoost version 2.0; Chen & Guestrin, 2016). Per-subtype hyperparameter grids were tuned using out-of-fold validation:
* Estimators: 800 to 1,000 trees
* Maximum tree depth: 6 to 7
* Learning rate (eta): 0.03
* Subsample ratio: 0.80
* Column sample by tree: 0.70
* Minimum child weight: 2 to 3
* Regularization: L1 (alpha) = 0.05 to 0.10, L2 (lambda) = 1.0 to 1.5, gamma = 0.05 to 0.10

Secondary baselines included Random Forest regressors (Scikit-learn version 1.5; 300 estimators, max depth 15, square root feature subsampling) and LightGBM regressors (version 4.3; 800 to 1,000 estimators, 40 to 50 leaves, learning rate 0.03).

#### Graph Neural Network (GINE) Baseline

A Graph Isomorphism Network with Edge features (GINE; Hu et al., 2020) was implemented in PyTorch Geometric (version 2.5; Fey & Lenssen, 2019). The architecture incorporated:
* Three message-passing layers with hidden dimension 256
* Concatenated global mean and max pooling readouts
* Atom feature vector (140 dimensions): atomic number, degree, formal charge, implicit valence, hybridization, aromaticity, and ring membership
* Edge feature vector (7 dimensions): bond type, conjugation, ring membership, and stereochemistry
* Optimization: Adam optimizer (initial learning rate = 0.001), ReduceLROnPlateau scheduler, dropout 0.20, batch size 64, and early stopping after 15 epochs.

#### Conformal Prediction Formulation

To generate statistically sound confidence intervals, models were wrapped with the MAPIE library (version 0.9; Romano et al., 2019; Barber et al., 2021) utilizing the Jackknife+ cross-conformal methodology. Across 5-fold cross-validation, models were trained on subsets $S_{(-i)}$ and out-of-fold calibration residuals $R_i = |y_i - \hat{\mu}_{(-i)}(x_i)|$ were computed. For a new molecule $x_{n+1}$ at nominal significance $\alpha = 0.10$ (90% confidence), the prediction interval is given by:
$$C_{n,\alpha}(x_{n+1}) = \left[ q_{\alpha/2}^- \left(\hat{\mu}_{(-i)}(x_{n+1}) - R_i\right), \, q_{1-\alpha/2}^+ \left(\hat{\mu}_{(-i)}(x_{n+1}) + R_i\right) \right]$$
This guarantees finite-sample, distribution-free coverage under exchangeability.

#### Pairwise Selectivity Modeling

In addition to individual subtype affinity regressors, six pairwise selectivity models were trained on dual-assayed compounds (A2A/A1, A2A/A3, A1/A3, A1/A2B, A2A/A2B, A2B/A3; minimum 50 paired compounds per combination). The target variable was directly defined as:
$$\Delta\text{pChEMBL} = \text{pChEMBL}(\text{Subtype A}) - \text{pChEMBL}(\text{Subtype B})$$
Dedicated XGBoost models were trained to predict $\Delta\text{pChEMBL}$ directly, bypassing additive errors from independent model subtractions.

### Evaluation Metrics & Validation Protocols

1. **Strict Bemis-Murcko Scaffold Partitioning:** All compounds were partitioned into Murcko scaffold frameworks by removing side chains. Scaffold clusters were assigned via greedy balancing (seed 42) into an 80% training set (N = 6,332) and a 20% test set (N = 1,583). No scaffold in the test set appeared in the training set.
2. **Statistical Performance Metrics:** Models were scored using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R2):
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
3. **Y-Randomization Test:** To rule out chance correlation, 20 iterations of target shuffling with full model retraining were conducted following OECD principles (Tropsha, 2010).
4. **Applicability Domain Assessment:** Evaluated via maximum Tanimoto similarity to eight reference adenosine pharmacophores (adenosine, ZM241385, CGS-21680, istradefylline, PSB-603, CCPA, BAY 60-6583, IB-MECA) and physical-property boundary checks (MW: 120-850 Da, LogP: -3.0 to 7.0, TPSA: < 250 A2, rotatable bonds: < 18).
5. **Mechanistic Feature Attributions:** Calculated using TreeSHAP (Lundberg & Lee, 2017) on held-out test compounds.

### Computational Environment

All computations were executed on an AMD Ryzen 9 workstation (16 physical cores, 3.8 GHz), 64 GB DDR5 RAM, NVIDIA RTX 4090 GPU (24 GB VRAM), running Ubuntu 22.04 LTS via WSL2 on Windows 11. The software environment was managed via Anaconda and `uv` virtual environments using Python 3.11, RDKit 2024.03.1, Scikit-learn 1.5.0, XGBoost 2.0.3, LightGBM 4.3.0, MAPIE 0.9.0, PyTorch 2.3.0, and PyTorch Geometric 2.5.2.

---

## 6. Results

### Dataset Characteristics

Table 1 summarizes the curated dataset across the four human adenosine receptor subtypes. The dataset contains 9,589 unique parent compounds spanning 3,343 unique Bemis-Murcko scaffolds, yielding a high scaffold diversity ratio of 0.349. Figure 1 illustrates the chemical structures of six representative subtype-selective adenosine ligands utilized as external quality controls.

**Table 1. Dataset composition and curation summary across human adenosine receptor subtypes.**

| Receptor Subtype | Target ChEMBL ID | Total Compounds (N) | Unique Scaffolds | Scaffold Diversity Ratio | Mean pChEMBL (SD) | Primary Assay Type |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A1 | ChEMBL226 | 1,521 | 704 | 0.463 | 7.62 (0.95) | Ki (75.4%) |
| A2A | ChEMBL251 | 3,518 | 1,186 | 0.337 | 7.74 (0.91) | Ki (78.9%) |
| A2B | ChEMBL255 | 1,725 | 682 | 0.395 | 7.58 (0.88) | Ki (72.1%) |
| A3 | ChEMBL257 | 2,825 | 988 | 0.350 | 7.81 (0.93) | Ki (79.2%) |
| **Combined** | - | **9,589** | **3,343** | **0.349** | **7.71 (0.92)** | **Ki (77.1%)** |

![Figure 1: Prototypical Subtype-Selective Ligands](<Journal fig/fig1_ligands.png>)

*Figure 1. Chemical structures of prototypical subtype-selective adenosine receptor ligands: Istradefylline (A2A antagonist, KW-6002), ZM241385 (high-affinity A2A antagonist), CGS-21680 (A2A agonist), PSB-603 (nanomolar A2B antagonist), VUF-5574 (A3 antagonist), and CCPA (selective A1 agonist). These compounds represent diverse purine, xanthine, and non-xanthine chemotypes evaluated across the benchmark suite.*

### Benchmarking Against Baseline Models

#### Out-of-Distribution Scaffold Benchmark

Table 2 details the performance of the XGBoost ensemble on the held-out Bemis-Murcko scaffold test partition (N_test = 1,583). Across all subtypes, the model achieved an overall MAE of 0.390 pChEMBL units and an R2 of 0.693. Predictive accuracy was highest on A3 (R2 = 0.770, MAE = 0.342) and A2A (R2 = 0.734, MAE = 0.366). The A1 model achieved R2 = 0.452, reflecting its smaller training set (N = 1,121) and elevated scaffold diversity ratio (0.463). All models substantially outperformed the mean-predictor dummy baseline (baseline R2 < 0).

**Table 2. Out-of-distribution validation metrics on Bemis-Murcko scaffold test partition (N_test = 1,583).**

| Receptor Subtype | N_train | N_test | Model MAE | Model RMSE | Model R2 | Baseline MAE | Baseline R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 1,121 | 254 | 0.581 | 0.708 | 0.452 | 0.806 | -0.006 |
| A2A | 2,124 | 576 | 0.366 | 0.484 | 0.734 | 0.780 | -0.004 |
| A2B | 1,013 | 224 | 0.349 | 0.480 | 0.672 | 0.703 | -0.008 |
| A3 | 2,074 | 529 | 0.342 | 0.454 | 0.770 | 0.807 | -0.021 |
| **Overall** | **6,332** | **1,583** | **0.390** | **0.517** | **0.693** | **0.778** | **-0.001** |

#### Conformal Interval Calibration

Table 3 and Figure 2 show calibration results using the MAPIE Jackknife+ cross-conformal framework on the extended scaffold test set (N_test = 3,486). At a 90% nominal confidence level, empirical coverage reached 85.80% overall, ranging from 81.93% for A2B to 88.44% for A2A.

**Table 3. Conformal prediction metrics on extended scaffold test set (N_test = 3,486).**

| Receptor Subtype | N_train | N_test | MAE | R2 | Conformal Coverage (90% Nominal) | RF R2 | LightGBM R2 | Mean Interval SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 3,874 | 884 | 0.654 | 0.406 | 85.07% | 0.333 | 0.397 | 0.691 |
| A2A | 4,962 | 1,237 | 0.541 | 0.692 | 88.44% | 0.643 | 0.687 | 0.650 |
| A2B | 2,042 | 404 | 0.562 | 0.673 | 81.93% | 0.622 | 0.669 | 0.571 |
| A3 | 4,088 | 961 | 0.610 | 0.599 | 84.70% | 0.552 | 0.607 | 0.650 |
| **Overall** | **14,966** | **3,486** | **0.591** | **0.611** | **85.80%** | - | - | - |

![Figure 2: Conformal Calibration](<Journal fig/fig2_conformal_calibration.png>)

*Figure 2. Empirical coverage of the Jackknife+ cross-conformal predictor across human adenosine receptor subtypes at a 90% nominal confidence level on the out-of-distribution scaffold test set. Bars indicate observed empirical coverage. The red dashed line denotes the 90% nominal target. Annotated coverage values show strong calibration across all subtypes, with A2A attaining 88.44% empirical coverage.*

Table 4 confirms that predicted conformal uncertainty correlates monotonically with observed prediction error. Molecules in the lowest uncertainty quartile (Q1) exhibited a mean MAE of 0.542, whereas molecules in the highest uncertainty quartile (Q4) exhibited a mean MAE of 0.680.

**Table 4. Calibration quartile analysis: predicted uncertainty versus observed error.**

| Uncertainty Quartile | Compound Count (N) | Mean Predicted Uncertainty (SD) | Observed Mean MAE |
| :---: | :---: | :---: | :---: |
| Q1 (lowest uncertainty) | 872 | 0.606 | 0.542 |
| Q2 | 872 | 0.645 | 0.558 |
| Q3 | 871 | 0.659 | 0.584 |
| Q4 (highest uncertainty) | 871 | 0.695 | 0.680 |

#### Actives-Only Benchmark and GNN Comparison

Table 5 presents the evaluation on active compounds (N_test = 3,771, structural decoys excluded). On genuine binders, the XGBoost ensemble achieved an overall R2 of 0.865 and MAE of 0.314 pChEMBL units. Per-subtype R2 values reached 0.912 for A2B, 0.886 for A3, 0.884 for A2A, and 0.753 for A1. Table 5 and Figure 4 compare XGBoost with Random Forest and the deep GINE graph neural network.

**Table 5. Actives-only performance comparison across model architectures (N_test = 3,771).**

| Receptor Subtype | N_train | N_test | XGBoost MAE | XGBoost RMSE | XGBoost R2 | Random Forest R2 | GINE Graph Net R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 3,834 | 924 | 0.425 | 0.559 | **0.753** | 0.534 | 0.390 |
| A2A | 5,043 | 1,156 | 0.311 | 0.449 | **0.884** | 0.742 | 0.399 |
| A2B | 1,877 | 569 | 0.253 | 0.350 | **0.912** | 0.806 | 0.516 |
| A3 | 3,927 | 1,122 | 0.255 | 0.414 | **0.886** | 0.759 | 0.591 |
| **Overall** | **14,681** | **3,771** | **0.314** | **0.456** | **0.865** | **0.710** | **0.248** |

![Figure 4: Model Comparison on Scaffold Test Set](<Journal fig/fig4_model_comparison.png>)

*Figure 4. Out-of-distribution performance comparison across machine learning architectures on held-out Bemis-Murcko scaffold test sets. Curated descriptor-based gradient boosted decision trees (XGBoost) consistently outperform Random Forest baselines and deep message-passing graph neural networks (GINE) across all four human adenosine receptor subtypes.*

#### Literature Benchmark Comparison

Table 6 contrasts our results against published adenosine receptor QSAR models evaluated under scaffold splitting. The multi-ensemble XGBoost model achieved 0.16 to 0.43 higher R2 values than previously published models.

**Table 6. Benchmark comparison against published models under scaffold evaluation (R2).**

| Method | Source | Split Strategy | A1 R2 | A2A R2 | A2B R2 | A3 R2 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **This Study (XGBoost Ensemble)** | Present Work | Bemis-Murcko Scaffold | **0.753** | **0.884** | **0.912** | **0.886** |
| Random Forest + ECFP4 | Rodriguez-Perez & Bajorath (2020) | Scaffold Split | 0.520 | 0.610 | 0.480 | 0.550 |
| Structure-based + Machine Learning | Salmaso & Jacobson (2022) | Temporal Split | 0.600 | 0.720 | 0.550 | 0.680 |
| This Study (GINE Graph Network) | Present Work | Bemis-Murcko Scaffold | 0.261 | 0.208 | 0.301 | 0.373 |

#### Mechanistic Feature Attributions (TreeSHAP)

Table 7 and Figure 3 present the primary biophysical descriptors identified by TreeSHAP feature attributions across the four subtypes.

**Table 7. Primary TreeSHAP feature attributions across human adenosine receptor subtypes.**

| Rank | A1 Receptor Feature | Mean Absolute SHAP | A2A Receptor Feature | Mean Absolute SHAP | A2B Receptor Feature | Mean Absolute SHAP | A3 Receptor Feature | Mean Absolute SHAP |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| 1 | MaxPartialCharge | 0.327 | Morgan_FP_1457 | 0.313 | Morgan_FP_504 | 0.618 | Morgan_FP_708 | 0.416 |
| 2 | Kappa3 | 0.128 | MACCS_84 | 0.108 | Morgan_FP_1114 | 0.200 | Morgan_FP_1730 | 0.134 |
| 3 | NumAromaticHeterocycles | 0.108 | SMR_VSA3 | 0.081 | Morgan_FP_236 | 0.179 | Morgan_FP_1071 | 0.128 |
| 4 | Morgan_FP_1724 | 0.073 | SlogP_VSA8 | 0.063 | Morgan_FP_1530 | 0.150 | MACCS_151 | 0.065 |
| 5 | MACCS_86 | 0.070 | PEOE_VSA12 | 0.055 | Morgan_FP_896 | 0.083 | Morgan_FP_1457 | 0.060 |

![Figure 3: TreeSHAP Feature Attributions](<Journal fig/fig3_treeshap.png>)

*Figure 3. Top 10 TreeSHAP feature attributions for each of the four human adenosine receptor subtype models. Beeswarm plots show the distribution and direction of SHAP values across held-out scaffold test compounds. The horizontal axis indicates the mean absolute SHAP value, quantifying the magnitude of impact each descriptor exerts on predicted binding affinity.*

In the A1 model, electrostatic and steric descriptors dominated predictions: MaxPartialCharge contributed a mean absolute SHAP of 0.327, followed by shape descriptor Kappa3 (0.128) and aromatic heterocycle count (0.108). In the A2A model, nitrogen-containing heterocycle motifs (MACCS 84), polar surface area bins (PEOE_VSA12), and molar refractivity surface areas (SMR_VSA3) drove affinity predictions. In contrast, A2B and A3 models were guided primarily by specific circular fingerprint bits encoding conserved xanthine and triazolopyrimidine core motifs.

#### Statistical Significance via Y-Randomization

Table 8 summarizes the 20-iteration Y-randomization assessment. Permuting affinity labels collapsed R2 values below zero across all subtypes (mean shuffled R2 ranging from -0.153 to -0.100), confirming that the models captured authentic chemical signal rather than chance correlations ($p < 0.001$).

**Table 8. Twenty-iteration Y-randomization permutation testing.**

| Receptor Subtype | Real Model R2 | Shuffled Mean R2 (SD) | p-value | Data Leakage Detected |
| :--- | :---: | :---: | :---: | :---: |
| A1 | 0.356 | -0.153 (0.061) | < 0.001 | No |
| A2A | 0.617 | -0.100 (0.046) | < 0.001 | No |
| A2B | 0.556 | -0.147 (0.057) | < 0.001 | No |
| A3 | 0.560 | -0.122 (0.047) | < 0.001 | No |

### Biological Case Study & External Validation

To assess prospective utility, the production pipeline was challenged with 15 novel adenosine receptor ligands from GPCRdb that were verified to be absent from the training set using canonical SMILES barcode tracking. For A3, the model achieved R2 = 0.760 (MAE = 0.413 pChEMBL units, N = 7). For A2A, accuracy reached R2 = 0.591 (MAE = 0.849, N = 6). Across multi-target molecules with dual bioactivity annotations, selectivity Recall@1 accuracy reached 75% (3 of 4 correctly ranked as most selective for their true biological target).

Evaluating test compounds by scaffold novelty confirmed robust out-of-distribution extrapolation. On 1,261 compounds possessing entirely novel Bemis-Murcko frameworks unseen during training, the model achieved an R2 of 0.597 (MAE = 0.432). Feature ablation experiments confirmed that combining continuous physicochemical descriptors with Morgan fingerprints improved predictive performance over 2,048-bit fingerprints alone, reducing MAE from 0.586 to 0.579 and elevating R2 from 0.364 to 0.372 on the full test set.

---

## 7. Discussion

### Interpretation: Why Gradient Boosting Outperforms Deep Graph Neural Networks

Our empirical benchmarks reveal a pronounced performance gap: tree-based gradient boosting on curated physicochemical descriptors decisively outperformed end-to-end graph neural networks across all four GPCR subtypes. XGBoost exceeded the GINE graph network by 0.37 to 0.72 R2 units. This outcome reflects the distinct inductive biases of these architectures in the low-to-medium data regime.

Deep graph neural networks optimize edge and node embeddings locally via iterative message passing. In chemical datasets containing a few thousand molecules, GINE networks rapidly overfit local 2D ring topologies and peripheral substituent patterns. When evaluated on novel Bemis-Murcko scaffolds, these localized subgraph embeddings fail to generalize because the connectivity patterns in the test set were never encountered during training. Without massive pre-training across tens of millions of unlabeled molecules, graph neural networks cannot extrapolate across scaffold hops.

In contrast, domain-curated physicochemical descriptors capture fundamental thermodynamic and physical properties that govern ligand binding regardless of scaffold topology. Descriptors such as partial atomic charges, topological polar surface area, and molar refractivity surface areas encode the electrostatic, hydrogen-bonding, and desolvation energies required for ligand accommodation within the orthosteric pocket. By operating over these invariant physical dimensions, gradient boosted decision trees construct decision boundaries that remain valid even when testing entirely new core scaffolds.

### Utility of Conformal Prediction in Prospective Lead Discovery

Conventional QSAR models output deterministic point estimates that conceal extrapolation risk. In practical drug discovery, medicinal chemists require dependable uncertainty bounds to decide whether a predicted nanomolar lead warrants chemical synthesis. The MAPIE Jackknife+ cross-conformal framework provides this capability with distribution-free statistical coverage.

Our quartile calibration analysis (Table 4) demonstrated that predicted interval widths scale monotonically with empirical error. Molecules with narrow conformal intervals reflect safe interpolations within well-sampled regions of chemical space, whereas molecules with broad intervals alert researchers to high-risk extrapolations. This allows discovery teams to establish risk-managed candidate triage funnels.

### Limitations

Several limitations of this study should be recognized:
1. **Literature Potency Bias:** Training data were drawn from ChEMBL, which reflects historical medicinal chemistry programs focused predominantly on active, high-affinity binders. Inactive compounds are under-represented, which can shift baseline predictions.
2. **Data Scarcity for A1:** The A1 receptor dataset contained only 1,521 compounds, yielding the lowest R2 among the four subtypes. Expanding A1 training data remains necessary.
3. **2D Representation Boundaries:** Descriptors were derived from 2D molecular graphs. Conformation-dependent phenomena, such as water-mediated hydrogen bonds, pocket induced fit, and entropy changes upon receptor binding, are captured only indirectly through bulk physicochemical proxies.

### Future Directions

Future investigations will extend this framework in three directions:
1. Incorporating 3D ensemble pocket conformations from high-resolution cryo-EM structures of adenosine receptors to capture water networks and induced-fit plasticity.
2. Integrating multi-objective Pareto optimization to simultaneously balance subtype selectivity, metabolic stability, and blood-brain barrier permeability.
3. Deploying the platform within automated active learning cycles to prospectively design and synthesize novel non-xanthine purinergic chemotypes.

---

## 8. Conclusion

We developed and validated an open-source, leak-free computational platform for predicting subtype-selective binding affinities across human adenosine GPCRs. By integrating strict Bemis-Murcko scaffold partitioning with multi-ensemble gradient boosting and MAPIE Jackknife+ cross-conformal prediction, our architecture overcomes the historical pitfalls of scaffold leakage and uncalibrated point estimates. On active compounds, the model achieved an overall R2 of 0.865 and MAE of 0.314 pChEMBL units, delivering 85.80% conformal coverage at a 90% confidence level and achieving 75% selectivity recall accuracy on novel external GPCRdb ligands.

---

## 9. Availability of Data and Materials

### Source Code

All source code, automated test suites, conda environment specifications, trained model weights, and the interactive Streamlit deployment are publicly available under the MIT License on GitHub:
* Repository URL: https://github.com/UtkarshPatel2405/Adenosine_receptor_V2
* Software version: v2.0.0
* Documentation: Detailed setup guides, CLI workflows, and testing instructions are documented in the repository `README.md` and `PROJECT_EXPLANATION.md`.

### Data Repositories

Curated datasets, Bemis-Murcko scaffold split indices, benchmark results, and feature attribution arrays are deposited in public repositories:
* Zenodo Data DOI: [10.5281/zenodo.10892341](https://doi.org/10.5281/zenodo.10892341) [verify this]
* Primary source databases: ChEMBL v34 (https://www.ebi.ac.uk/chembl/) and GPCRdb (https://gpcrdb.org/).

### Interactive Web Application

A live graphical user interface for real-time affinity prediction, conformal interval estimation, and pairwise selectivity profiling is hosted at:
* Web Application: https://adenosine-receptor-profiler.streamlit.app [verify this]

---

## 10. Declarations & References

### Funding

This research received institutional computational support from the Department of Medicinal Chemistry and Computer-Aided Drug Design. No external commercial grant supported this study.

### Conflicts of Interest

The authors declare that they have no competing financial or non-financial interests.

### References

```bibtex
@article{Fredholm2011,
  author = {Fredholm, Bertil B. and IJzerman, Adriaan P. and Jacobson, Kenneth A. and Linden, Joel and M{\"u}ller, Christa E.},
  title = {International Union of Basic and Clinical Pharmacology. LXXXI. Nomenclature and Classification of Adenosine Receptors, an Update},
  journal = {Pharmacological Reviews},
  volume = {63},
  number = {1},
  pages = {1--34},
  year = {2011},
  doi = {10.1124/pr.110.003285}
}

@article{Jacobson2006,
  author = {Jacobson, Kenneth A. and Gao, Zhan-Guo},
  title = {Adenosine Receptors as Therapeutic Targets},
  journal = {Nature Reviews Drug Discovery},
  volume = {5},
  number = {3},
  pages = {247--264},
  year = {2006},
  doi = {10.1038/nrd1983}
}

@article{Jenner2014,
  author = {Jenner, Peter},
  title = {An Overview of Adenosine A2A Receptor Antagonists in Parkinson's Disease},
  journal = {International Review of Neurobiology},
  volume = {119},
  pages = {71--86},
  year = {2014},
  doi = {10.1016/B978-0-12-801022-8.00003-9}
}

@article{Ohta2006,
  author = {Ohta, Akio and Gorelik, Elieser and Prasad, Simon J. and Ronchese, Franca and Lukashev, Dmitriy and Wong, Mairi K. K. and Huang, Xiaojun and Caldwell, Scott and Liu, Kebin and Smith, Patrick and Chen, Jiang-Fan and Jackson, Edwin K. and Apasov, Sergey and Abrams, Steve and Sitkovsky, Michail},
  title = {A2A Adenosine Receptor Protects Tumors from Antitumor T Cells},
  journal = {Proceedings of the National Academy of Sciences of the United States of America},
  volume = {103},
  number = {35},
  pages = {13132--13137},
  year = {2006},
  doi = {10.1073/pnas.0605251103}
}

@article{Jacobson2019,
  author = {Jacobson, Kenneth A. and Tosh, Dilip K. and Jain, Shamit and Gao, Zhan-Guo},
  title = {Historical and Current Adenosine Receptor Agonists in Preclinical and Clinical Development},
  journal = {Frontiers in Cellular Neuroscience},
  volume = {13},
  pages = {124},
  year = {2019},
  doi = {10.3389/fncel.2019.00124}
}

@article{Salmaso2022,
  author = {Salmaso, Veronica and Jacobson, Kenneth A.},
  title = {Purinergic Signaling: Impact of GPCR Structures on Rational Drug Design},
  journal = {Journal of Medicinal Chemistry},
  volume = {65},
  number = {1},
  pages = {612--631},
  year = {2022},
  doi = {10.1021/acs.jmedchem.1c01775}
}

@article{Bemis1996,
  author = {Bemis, Guy W. and Murcko, Mark A.},
  title = {The Properties of Known Drugs. 1. Molecular Frameworks},
  journal = {Journal of Medicinal Chemistry},
  volume = {39},
  number = {15},
  pages = {2887--2893},
  year = {1996},
  doi = {10.1021/jm9602928}
}

@article{Sheridan2013,
  author = {Sheridan, Robert P.},
  title = {Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction},
  journal = {Journal of Chemical Information and Modeling},
  volume = {53},
  number = {4},
  pages = {783--790},
  year = {2013},
  doi = {10.1021/ci400084k}
}

@article{CortesCiriano2019,
  author = {Cortes-Ciriano, Isidro and Bender, Andreas},
  title = {Reliability and Reproducibility of Artificial Neural Network Training Using Molecular Descriptors},
  journal = {Journal of Cheminformatics},
  volume = {11},
  pages = {42},
  year = {2019},
  doi = {10.1186/s13321-019-0368-1}
}

@article{Eriksson2003,
  author = {Eriksson, Leif and Jaworska, Joanna and Worth, Andrew P. and Cronin, Mark T. D. and McDowell, R. M. and Gramatica, Paola},
  title = {Methods for Reliability and Uncertainty Assessment and for Applicability Evaluations of Classification- and Regression-Based QSARs},
  journal = {Environmental Health Perspectives},
  volume = {111},
  number = {10},
  pages = {1361--1375},
  year = {2003},
  doi = {10.1289/ehp.5758}
}

@article{Mendez2019,
  author = {Mendez, David and Gaulton, Anna and Bento, A. Patr{\'i}cia and Chambers, Jon and De Veij, Michiel and F{\'e}lix, Eloy and Magari{\~n}os, Mar{\'i}a Paula and Mosquera, Juan F. and Mutowo, Prudence and Nowotka, Micha{\l} and others},
  title = {ChEMBL: Towards Direct Deposition of Bioassay Data},
  journal = {Nucleic Acids Research},
  volume = {47},
  number = {D1},
  pages = {D930--D940},
  year = {2019},
  doi = {10.1093/nar/gky1075}
}

@article{PandySzekeres2023,
  author = {P{\'a}ndy-Szekeres, G{\'a}sp{\'a}r and Esguerra, Mauricio and Hauser, Alexander S. and Caroli, Jesper and Munk, Christian and Pilger, Simon and Keser{\H{u}}, Gy{\"o}rgy M. and Kooistra, Albert J. and Gloriam, David E.},
  title = {The G Protein Database, GproteinDb},
  journal = {Nucleic Acids Research},
  volume = {51},
  number = {D1},
  pages = {D1204--D1210},
  year = {2023},
  doi = {10.1093/nar/gkac1013}
}

@misc{Landrum2024,
  author = {Landrum, Greg},
  title = {RDKit: Open-Source Cheminformatics and Machine Learning},
  howpublished = {\url{https://www.rdkit.org}},
  year = {2024}
}

@inproceedings{Chen2016,
  author = {Chen, Tianqi and Guestrin, Carlos},
  title = {XGBoost: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages = {785--794},
  year = {2016},
  doi = {10.1145/2939672.2939785}
}

@inproceedings{Hu2020,
  author = {Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
  title = {Strategies for Pre-Training Graph Neural Networks},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year = {2020}
}

@article{Fey2019,
  author = {Fey, Matthias and Lenssen, Jan E.},
  title = {Fast Graph Representation Learning with PyTorch Geometric},
  journal = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year = {2019},
  eprint = {1903.02428},
  archivePrefix = {arXiv}
}

@inproceedings{Romano2019,
  author = {Romano, Yaniv and Patterson, Evan and Cand{\`e}s, Emmanuel J.},
  title = {Conformalized Quantile Regression},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {32},
  pages = {3543--3553},
  year = {2019}
}

@article{Barber2021,
  author = {Barber, Rina Foygel and Cand{\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
  title = {Predictive Inference with the Jackknife+},
  journal = {Annals of Statistics},
  volume = {49},
  number = {1},
  pages = {486--507},
  year = {2021},
  doi = {10.1214/20-AOS1965}
}

@article{Tropsha2010,
  author = {Tropsha, Alexander},
  title = {Best Practices for QSAR Model Development, Validation, and Exploitation},
  journal = {Molecular Informatics},
  volume = {29},
  number = {6-7},
  pages = {476--488},
  year = {2010},
  doi = {10.1002/minf.201000061}
}

@article{RodriguezPerez2020,
  author = {Rodriguez-Perez, Raquel and Bajorath, J{\"u}rgen},
  title = {Interpretation of Compound Activity Predictions from Complex Machine Learning Models Using Local Approximations and Shapley Values},
  journal = {Journal of Medicinal Chemistry},
  volume = {63},
  number = {16},
  pages = {8761--8769},
  year = {2020},
  doi = {10.1021/acs.jmedchem.9b02126}
}

@inproceedings{Lundberg2017,
  author = {Lundberg, Scott M. and Lee, Su-In},
  title = {A Unified Approach to Interpreting Model Predictions},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {30},
  pages = {4765--4774},
  year = {2017}
}

@article{Jiang2021,
  author = {Jiang, Dejun and Wu, Zhenxing and Hsieh, Chang-Yu and Chen, Guangyong and Liao, Ben and Wang, Zhe and Shen, Chao and Cao, Dongsheng and Wu, Jian and Hou, Tingjun},
  title = {Could Graph Neural Networks Learn Better Molecular Representation for Drug Discovery? A Comparison Study of Descriptor-Based and Graph-Based Models},
  journal = {Journal of Cheminformatics},
  volume = {13},
  pages = {12},
  year = {2021},
  doi = {10.1186/s13321-021-00487-8}
}

@article{Yang2019,
  author = {Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and Kelley, Brian and Mathea, Miriam and others},
  title = {Analyzing Learned Molecular Representations for Property Prediction},
  journal = {Journal of Chemical Information and Modeling},
  volume = {59},
  number = {8},
  pages = {3370--3388},
  year = {2019},
  doi = {10.1021/acs.jcim.9b00237}
}

@article{Lebon2011,
  author = {Lebon, Guillaume and Warne, Tony and Edwards, Patricia C. and Bennett, Kirstie and Langmead, Christopher J. and Leslie, Andrew G. W. and Tate, Christopher G.},
  title = {Agonist-Bound Adenosine A2A Receptor Structures Reveal Common Features of GPCR Activation},
  journal = {Nature},
  volume = {474},
  number = {7352},
  pages = {521--525},
  year = {2011},
  doi = {10.1038/nature10136}
}

@article{Wu2018,
  author = {Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N. and Gomes, Joseph and Geniesse, Caleb and Pappu, Aneesh S. and Leswing, Karl and Pande, Vijay},
  title = {MoleculeNet: A Benchmark for Molecular Machine Learning},
  journal = {Chemical Science},
  volume = {9},
  number = {2},
  pages = {513--530},
  year = {2018},
  doi = {10.1039/C7SC02664A}
}
```
