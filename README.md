# 🧬 Adenosine Receptor Profiler (v2.4.0)

**Industrial CADD & Conformal AI Platform for Multi-Target GPCR Selectivity, Functional MoA, Safety, and Structural Pocket Biology**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2024.03+-00C7B7.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MAPIE Conformal](https://img.shields.io/badge/Conformal-MAPIE%2090%25-8B5CF6.svg)](https://mapie.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live Web Application

Access the production deployment of the platform:

* 🚀 The live web link: https://adenosinereceptorv2-b45nnjh6s9u5xrpte7wcl4.streamlit.app/

---

## 🔬 Platform Architecture & Scientific Capabilities

Human adenosine receptors (**A₁, A₂A, A₂B, and A₃**) belong to the class A rhodopsin-like G protein-coupled receptor (GPCR) superfamily. Because their 7-transmembrane (7-TM) orthosteric pockets share **>70% sequence homology**, designing subtype-selective ligands without off-target liabilities is a central challenge in computer-aided drug discovery (CADD).

This platform delivers an end-to-end computational screening and pharmacology profiling suite:

```
                                  [ Input: Query Molecule SMILES ]
                                                  │
                                  ┌───────────────┴───────────────┐
                                  ▼                               ▼
                      [ RDKit Feature Engine ]        [ OECD Principle 3 AD ]
                      • 2048-bit Morgan FP            • Tanimoto scaffold proximity
                      • 167 MACCS Keys Frag           • MW, LogP, TPSA, RotB gates
                      • Curated 41 Physicochemical
                                  │                               │
                                  └───────────────┬───────────────┘
                                                  ▼
                              [ Multi-Model Quad-Tree Ensemble ]
                              • XGBoost + LightGBM + Random Forest
                              • MAPIE 5-Fold Cross-Conformal Bounds
                              • 7-TM Joint Covariance Regularization
                                                  │
                ┌───────────────────┬─────────────┴─────┬───────────────────┐
                ▼                   ▼                   ▼                   ▼
      [ Subtype Affinity ]  [ Functional MoA ]  [ Safety Liabilities ]  [ ADMET / CNS-MPO ]
      • pChEMBL & Ki (nM)   • Full/Partial Ag   • A1 Bradycardia        • Pfizer 6-Param MPO
      • ΔpChEMBL Deltas     • Antagonist / Inv  • A3 Mast Cell Risk     • Blood-Brain Barrier
      • Rank Hierarchy      • Gs vs Gi Cascade  • PDE1-10 Cross-Rx      • LogBB Permeability
```

---

### 1. Multi-Target Affinity & Thermodynamic $K_i$ Conversion
* **Subtype Spectrum**: Predicts binding affinity ($p\text{ChEMBL} = -\log_{10} K_i [\text{M}]$) simultaneously across all 4 human subtypes.
* **Thermodynamic Conversion**: Converts logarithmic $p\text{ChEMBL}$ into physiological equilibrium dissociation constants ($K_i$ in $\text{nM}$, $\mu\text{M}$, or $\text{pM}$).
* **Multi-Task 7-TM Covariance**: Applies an orthosteric covariance matrix to capture structural coupling and reduce orthogonal prediction noise across homologous pockets.

### 2. Finite-Sample Adaptive Conformal Uncertainty (90% Confidence)
* **Distribution-Free Guarantees**: Calibrated with MAPIE 5-fold cross-conformal Jackknife+ inference.
* **Scaffold-Adaptive Heteroscedastic Scaling**: Dynamically scales interval width based on topological distance to training chemotypes (core scaffolds vs scaffold hops).

### 3. Functional Mode of Action (MoA) & G-Protein Signaling
* **Efficacy Classification**: Structural SMARTS classification identifying ribose/carboxamide activation motifs (Full Agonist / Partial Agonist) versus xanthine/triazolopyrimidine cores (Neutral Antagonist / Inverse Agonist).
* **Signaling Cascade Profiling**: Maps downstream effectors:
  * $A_{2A} / A_{2B}$: $G_s$-coupled $\rightarrow$ Adenylyl Cyclase activation $\rightarrow$ $\text{cAMP}\uparrow$ (Vasodilation, Immunosuppression).
  * $A_1 / A_3$: $G_{i/o}$-coupled $\rightarrow$ Adenylyl Cyclase inhibition $\rightarrow$ $\text{cAMP}\downarrow$ (AV node slowing, Analgesia).

### 4. 3D Orthosteric Pocket Biology & Toggle Switches
* **Conserved Hydrogen-Bond Anchor**: Audits bidentate H-bonding to $\text{Asn}^{6.55}$ ($\text{Asn253}$ in $A_{2A}$, $\text{Asn254}$ in $A_1$, $\text{Asn250}$ in $A_3$).
* **Transmission Toggle Switch**: Evaluates steric engagement with $\text{Trp}^{6.48}$ triggering the outward swing of TM6.
* **Chiral Stereocenter Audit**: Flags natural D-ribose vs L-ribose stereochemical potency cliffs ($>1000\times$ activity shift).

### 5. Cardiac & Anti-Target Safety Panel
* **$A_1$ Bradycardia Liability**: Alerts for sub-nanomolar $A_1$ off-target potency associated with atrioventricular (AV) block.
* **$A_3$ Mast Cell Degranulation**: Quantifies histamine release and bronchoconstriction risk.
* **PDE1–10 Cross-Reactivity**: RDKit SMARTS screening for xanthine/purine cross-inhibition of phosphodiesterases.

### 6. Pfizer CNS-MPO & Blood-Brain Barrier (BBB) Permeability
* **Pfizer 6-Parameter Multi-Parameter Optimization (CNS-MPO)**: Calculates composite score ($0.0 - 6.0$) from $\text{CLogP}$, $\text{CLogD}$, $\text{MW}$, $\text{TPSA}$, $\text{HBD}$, and $\text{p}K_a$.
* **Clark's LogBB**: Predicts brain-to-plasma partitioning ($\log BB = 0.152\cdot\text{CLogP} - 0.0148\cdot\text{TPSA} + 0.139$).

---

## 📊 Benchmark Metrics (Out-of-Distribution Scaffold Test Sets)

Models were validated strictly on held-out **Bemis–Murcko scaffold partitions** (no training scaffold overlap with test sets):

| Receptor Subtype | Test Samples ($n$) | Test $R^2$ | MAE (log units) | RMSE | 90% Conformal Coverage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Human $A_1$** | 884 | **0.692** | **0.385** | 0.512 | 91.2% |
| **Human $A_{2A}$** | 1,237 | **0.718** | **0.362** | 0.485 | 92.0% |
| **Human $A_{2B}$** | 404 | **0.684** | **0.395** | 0.528 | 89.8% |
| **Human $A_3$** | 961 | **0.675** | **0.410** | 0.542 | 91.5% |
| **Global Ensemble** | **3,486** | **0.693** | **0.390** | **0.517** | **91.2%** |

### Statistical Null Controls & Interpretability
* **Y-Randomization Tests (20 Permutations)**: Real model $R^2 = 0.693$ collapsed to shuffled null $R^2 = -0.102 \pm 0.041$, ruling out chance correlation.
* **Global TreeSHAP Feature Attributions**: Confirms heavy dependence on pharmacophore descriptors ($\text{MolLogP}$, $\text{TPSA}$, $\text{HBD}$, aromatic carbocycles, partial charges) matching GPCR crystallographic binding pocket chemistry.

---

## 🖥️ Streamlit 2.0 Web Application Layout

The graphical dashboard is organized into specialized workspaces:

1. **Single Molecule Profiler**:
   * **Tab 1: Overview & $K_i$**: Primary 4-subtype affinity table, experimental ground-truth alignment, multi-model consensus, CSV/JSON/SDF export.
   * **Tab 2: 2D/3D Conformer**: RDKit 2D depiction & interactive MMFF94 energy-minimized 3D conformer viewer (`3Dmol.js`).
   * **Tab 3: 4-Subtype Selectivity**: Multilateral polar radar polygon and complete pairwise $\Delta p\text{ChEMBL}$ differential matrix.
   * **Tab 4: Mode of Action (MoA)**: Efficacy classification, $G$-protein cascade diagram, and $\text{Asn}^{6.55}/\text{Trp}^{6.48}$ pocket engagement.
   * **Tab 5: Safety & CNS-MPO**: $A_1$ cardiac bradycardia, $A_3$ mast cell liability, PDE cross-reactivity, and Pfizer CNS-MPO BBB score.
   * **Tab 6: Drug-Likeness (QED)**: Quantitative Estimation of Drug-likeness, Lipinski Rule-of-5 compliance, and PAINS substructure liability filter.
   * **Tab 7: Chemical Space & Neighbors**: Nearest training-set neighbors by Tanimoto similarity with interactive 2D bubble scatter.
   * **Tab 8: Explainable AI (TreeSHAP)**: Local Shapley additive explanations breaking down per-feature contributions to subtype binding.
   * **Tab 9: Pocket Biology & GPCRdb**: Structural analog matching and interactive 3D co-crystal complex explorer (`3Dmol.js`).
   * **Tab 10: Provenance Audit**: SHA-256 cryptographic digests verifying model binaries, scalers, and reproducible datasets.
2. **Batch Virtual Screening**: High-throughput library screening with automated numeric CSV downloads.
3. **Model Benchmark Suite**: Publication-grade performance charts, conformal quartile calibrations, TreeSHAP bar plots, and dataset downloads.
4. **Structural Biology 3D Gallery**: Curated active (agonist-bound) vs inactive (antagonist-bound) crystallographic complexes from RCSB PDB & GPCRdb:
   * **$A_1$**: `6D9H` (Cryo-EM, 3.6 Å) vs `5N2S` (X-ray, 3.3 Å)
   * **$A_{2A}$**: `6GDG` (Cryo-EM, 2.6 Å) vs `4EIY` (X-ray, 1.8 Å)
   * **$A_{2B}$**: `6LPJ` (Cryo-EM, 3.2 Å) vs `8JZX` (Cryo-EM, 3.1 Å)
   * **$A_3$**: `7VAK` (Cryo-EM, 3.0 Å) vs `8HN0` (Cryo-EM, 3.2 Å)

---

## 📦 Quickstart & Installation

### Option 1: UV Package Manager (Recommended — Fast)

```bash
# Clone the repository
git clone https://github.com/UtkarshPatel2405/Adenosine_receptor_V2.git
cd Adenosine_receptor_V2

# Create virtual environment and sync dependencies
uv venv
uv pip install -r requirements.txt

# Launch Streamlit web dashboard
uv run streamlit run streamlit_app.py
```

### Option 2: Standard Python & Pip

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Launch application
streamlit run streamlit_app.py
```

### Option 3: Docker Deployment

```bash
# Build Docker image
docker build -t adenosine-profiler .

# Run container on port 8501
docker run -p 8501:8501 adenosine-profiler
```

---

## 📂 Repository Structure

```
Adenosine_receptor_V2/
├── streamlit_app.py           # Streamlit 2.0 Web Application Entrypoint
├── Dockerfile                 # Containerization & Hugging Face deployment
├── pyproject.toml             # Package build specification & metadata
├── requirements.txt           # Production Python dependency lock
├── src/                       # Core Architecture & Analytical Modules
│   ├── predictor.py           # Master inference orchestrator
│   ├── features.py            # 2048-bit Morgan, MACCS, & curated 41 descriptors
│   ├── chem_utils.py          # RDKit 2D SVG, MMFF94 3D conformers, PAINS, QED
│   ├── config.py              # Central hyperparameters, paths, and run IDs
│   ├── data_loader.py         # Curated ChEMBL/GPCRdb loading & deduplication
│   ├── applicability_domain.py# OECD Principle 3 quantitative domain gates
│   ├── provenance.py          # Deterministic SHA-256 cryptographic hashing
│   ├── models/                # Core Pharmacology & Inference Engines
│   │   ├── model_loader.py    # Model, scaler, and db_lookup caching loader
│   │   ├── ensemble_engine.py # Quad-tree inference & conformal boundaries
│   │   ├── selectivity_engine.py # Pairwise differential matrix (ΔpChEMBL)
│   │   ├── efficacy_engine.py # Functional MoA (Agonist vs Antagonist)
│   │   ├── interaction_engine.py # 3D pocket anchors (Asn6.55, Trp6.48, Phe168)
│   │   ├── safety_engine.py   # Cardiac AV block & PDE off-target safety
│   │   ├── admet_engine.py    # Pfizer CNS-MPO & Blood-Brain Barrier (LogBB)
│   │   ├── adaptive_conformal.py # Scaffold-adaptive conformal calibration
│   │   └── multitask_covariance.py # 7-TM orthosteric joint covariance & Ki
│   └── ui/                    # Streamlit Modern UI Components & Tabs
│       ├── styles.py          # Custom CSS, dark theme, and typography
│       ├── presets.py         # Canonical drug presets & GPCRdb records
│       ├── renderers_3d.py    # 3Dmol.js conformer & complex renderers
│       ├── tab_overview.py    # Tab 1: Executive summary & 4-subtype grid
│       ├── tab_structure.py   # Tab 2: 2D/3D conformer coordinates
│       ├── tab_selectivity.py # Tab 3: Selectivity radar polygon & deltas
│       ├── tab_efficacy.py    # Tab 4: Functional MoA & toggle switches
│       ├── tab_safety.py      # Tab 5: Safety panel & Pfizer CNS-MPO
│       ├── tab_druglikeness.py# Tab 6: QED & PAINS substructure filters
│       ├── tab_neighbors.py   # Tab 7: Nearest training neighbors scatter
│       ├── tab_shap.py        # Tab 8: TreeSHAP local feature explanations
│       ├── tab_structural.py  # Tab 9: GPCRdb pocket biology & complexes
│       ├── tab_provenance.py  # Tab 10: SHA-256 provenance audit trail
│       ├── tab_batch.py       # Batch virtual screening pipeline
│       ├── tab_benchmarks.py  # Model Benchmark Suite & publication metrics
│       └── tab_gallery.py     # Structural Biology 3D Gallery
├── data/                      # Dataset Repositories (ChEMBL & GPCRdb)
│   ├── raw/                   # Raw bioactivity CSVs
│   └── processed/             # Cleaned lookup dictionaries & training sets
├── models/                    # Trained Production Models (XGBoost, LGB, RF, Scaler)
├── outputs/                   # Benchmark reports, calibration curves, SHAP JSONs
└── tests/                     # Automated pytest test suites
```

---

## 📜 Citation & License

If you use this platform, models, or methodology in your computational drug discovery research, please cite:

```bibtex
@article{Patel2026AdenosineProfiler,
  title={Multi-Target Conformal Profiling and Selectivity Mapping Across Human Adenosine GPCRs with Scaffold-Adaptive Uncertainty Bounds},
  author={Patel, Utkarsh},
  year={2026},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{https://github.com/UtkarshPatel2405/Adenosine_receptor_V2}}
}
```

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
