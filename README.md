# Adenosine Receptor Profiler

Multi-target QSAR and cheminformatics platform for predicting equilibrium binding affinity (`pKi` and derived `Ki`), subtype selectivity, conformal uncertainty bounds, and applicability domain across the four human adenosine GPCR subtypes (`A1`, `A2A`, `A2B`, `A3`).

## Overview

Human adenosine receptors (`A1`, `A2A`, `A2B`, `A3`) are Class A GPCRs with conserved orthosteric binding pockets. This repository combines scaffold-split 3-model QSAR ensembles trained on curated human equilibrium binding data with deterministic RDKit physicochemical and SMARTS profiling modules served via a FastAPI and HTML/JS web application.

### Pure `Ki` / `Kd` (`pKi`) Data Curation

- **Sources**: Combines GPCRdb ligand exports (`data/raw/GPCRdb_*.xlsx`, aggregating ChEMBL and PDSP KiDatabase measurements) with human ChEMBL binding records (`CHEMBL226`, `CHEMBL251`, `CHEMBL255`, `CHEMBL256`).
- **Standardization**: Strictly retains experimental equilibrium binding constants (`Ki` and `Kd`, converted to `pKi = -log10(M)`), excluding functional/mixed potency endpoints and using zero synthetic decoys.
- **Structure Hygiene**: RDKit salt-stripping (largest organic fragment), charge neutralization, canonical SMILES generation, and median `pKi` deduplication per unique canonical structure and subtype.
- **Processed Outputs**: Four subtype-specific training tables (`data/processed/A1_pki.csv`, `data/processed/A2A_pki.csv`, `data/processed/A2B_pki.csv`, `data/processed/A3_pki.csv`).

### Predictive Modeling & Uncertainty Quantification

- **Molecular Featurization (`src/features.py`)**: 2048-bit Morgan fingerprints (radius 2), 167-bit MACCS keys, and RDKit physicochemical descriptors filtered by missingness, variance, and pairwise correlation (`FeatureFilter`) and standardized (`StandardScaler`).
- **3-Model Subtype Ensemble (`src/retrain_production.py`, `src/models/ensemble_engine.py`)**: Per-subtype consensus of **XGBoost**, **LightGBM**, and **Random Forest** regressors trained to predict `pKi`, with equilibrium conversion to `Ki` (`pM`, `nM`, or `uM`).
- **Conformal Prediction Intervals (`MAPIE`)**: 5-fold cross-conformal Jackknife+ calibration (`CrossConformalRegressor`, nominal 90% coverage, `alpha = 0.10`) providing finite-sample lower and upper affinity bounds.
- **Validation Protocol**: Out-of-distribution Bemis-Murcko scaffold split evaluated with `R2`, `RMSE`, `MAE`, and `Spearman rho`, alongside Y-randomization controls and TreeSHAP feature attribution (`src/shap_analysis.py`, `src/analysis.py`).
- **OECD Principle 3 Applicability Domain (`src/applicability_domain.py`, `src/models/adaptive_conformal.py`)**: Maximum 2048-bit Morgan Tanimoto similarity check against the curated training fingerprints in `data/processed/train_fps.pkl` paired with physicochemical property bounds and distance-aware interval scaling.
- **Selectivity & Provenance (`src/models/selectivity_engine.py`, `src/provenance.py`)**: Pairwise `Delta pKi` selectivity ratios across all four subtypes and SHA-256 artifact provenance tracking.

### Drug-Likeness, ADMET & 2D SMARTS Heuristics

The following modules are deterministic RDKit calculations, empirical formulas, or 2D SMARTS rule screens rather than models trained on experimental efficacy or toxicity endpoints:

- **Drug-Likeness & ADMET (`src/models/admet_engine.py`, `src/chem_utils.py`)**: Lipinski Rule-of-5, **Veber's Rules** (`TPSA <= 140 A^2` and `Rotatable Bonds <= 10`), PAINS structural alerts, RDKit QED, Pfizer 6-parameter CNS-MPO (`0.0 - 6.0`), Clark's empirical `LogBB`, and MMFF94 3D conformer generation.
- **2D SMARTS Heuristics (clearly labeled as rule-based screens)**:
  - **Efficacy Motif Hints (`src/models/efficacy_engine.py`)**: 2D SMARTS matching for ribose/carboxamide agonist motifs versus xanthine/triazolopyrimidine antagonist cores, mapped to canonical `Gs` (`A2A`, `A2B`) or `Gi/o` (`A1`, `A3`) coupling descriptions.
  - **Orthosteric Pharmacophore Screen (`src/models/interaction_engine.py`)**: 2D SMARTS screening for `Asn6.55` hydrogen-bonding motifs, `Trp6.48` purine-ribose toggle motifs, aromatic ring count (`Phe168` pi-stacking), and stereocenter annotations.
  - **Cardiac & Anti-Target Safety Alerts (`src/models/safety_engine.py`)**: Rule-based potency thresholds on predicted `A1` and `A3` affinity paired with 2D SMARTS screening for xanthine/purine scaffolds linked to phosphodiesterase (`PDE`) cross-reactivity.
- **Reference Structures (`src/presets.py`)**: Curated reference ligand presets and GPCRdb / RCSB PDB receptor structure metadata rendered interactively in the web client.

## Installation (Python 3.12)

Python `3.12` is required (`>=3.12,<3.14`).

### Option A: Conda (Recommended)

#### 1. Create environment directly from `environment.yml`

```bash
conda env create -f environment.yml && conda activate adenosine-profiler && python app.py
```

#### 2. Step-by-step `conda` + `pip` setup

Installing `rdkit` from `conda-forge` avoids native binary compilation issues on Python 3.12:

```bash
conda create -n adenosine-profiler -c conda-forge python=3.12 rdkit -y && conda activate adenosine-profiler && pip install -r requirements.txt && python app.py
```

### Option B: Python `venv` + `pip`

```bash
python -m venv .venv

# Windows (PowerShell or cmd)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

To install optional training and development extras (`optuna`, `pytest`, `ruff`) from `pyproject.toml`:

```bash
pip install -e ".[train,dev]"
```

## Accessing & Sharing the Web App

Start the FastAPI server with:

```bash
python app.py
```

- **Local URL**: `http://localhost:8000`
- **Shareable public HTTPS link from localhost (port forwarding)**:
  1. **Cloudflare Quick Tunnel (no account required)**:
     ```bash
     cloudflared tunnel --url http://localhost:8000
     ```
  2. **VS Code Ports panel**: Open the **Ports** tab in VS Code, forward port `8000`, and set its visibility to **Public**.

## Benchmark Metrics (Out-of-Distribution Scaffold Split)

Metrics below reflect the Bemis-Murcko scaffold split evaluation (`outputs/evaluation_report.json`) for the pure `pKi` 3-model ensemble (`XGBoost + LightGBM + Random Forest`).

<!-- BENCHMARKS_TABLE -->
All metrics below were generated from a single reproducible run (`run_id: ADENO_20261119_150621_4be4fd`) using **4 independent per-subtype feature scalers** (`models/precise/scaler_A1.pkl`, `scaler_A2A.pkl`, `scaler_A2B.pkl`, `scaler_A3.pkl`, each fitted strictly on that subtype's training split) and **5-fold out-of-fold validation-weighted consensus** (`models/precise/consensus_weights.json`) on `44,880` merged raw equilibrium binding (`Ki`/`Kd`) records (`GPCRdb`: `19,357`; `ChEMBL API`: `18,002`; `ChEMBL Parents`: `7,521`), deduplicated after salt-stripping, charge neutralization, and IQR-aware conflict filtering (`106` discordant groups dropped where `n = 2` with `spread > 2.0` or `n >= 3` with `IQR > 1.5`) into `16,355` unique `(molecule, subtype)` pairs across `9,015` unique molecules (`3,272` Bemis-Murcko scaffolds; `4,674` multi-subtype ligands; `3,853` weak/inactive entries with `pKi < 6.0`; `KI = 16,158`, `KD = 197`), partitioned strictly `80/20` with zero scaffold overlap (`13,082` train / `3,273` test records). In addition, **Model 2 (Use Case A: Functional Activity `pIC50 / pEC50`)** is trained independently on human `IC50`/`EC50` records across `data/processed/{A1,A2A,A2B,A3}_functional.csv` with its own 4 per-subtype scalers (`models/functional/scaler_{subtype}.pkl`).

| Target / Scope | Train N | Test N | Consensus R2 (XGB / LGB / RF) | Spearman rho (XGB / LGB / RF) | Consensus MAE | Consensus RMSE | 90% Coverage | Y-Rand Shuffled Mean R2 |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Human A1 (`A1_pki.csv`) | 3,686 | 890 | 0.477 (0.492 / 0.490 / 0.334) | 0.711 (0.707 / 0.708 / 0.679) | 0.611 | 0.785 | 86.0% | -0.081 (p = 0.048) |
| Human A2A (`A2A_pki.csv`) | 4,305 | 1,066 | 0.647 (0.659 / 0.650 / 0.558) | 0.809 (0.812 / 0.805 / 0.771) | 0.550 | 0.716 | 87.4% | -0.090 (p = 0.048) |
| Human A2B (`A2B_pki.csv`) | 1,531 | 471 | 0.725 (0.733 / 0.727 / 0.675) | 0.837 (0.838 / 0.835 / 0.816) | 0.522 | 0.653 | 88.1% | -0.171 (p = 0.048) |
| Human A3 (`A3_pki.csv`) | 3,560 | 846 | 0.585 (0.593 / 0.581 / 0.492) | 0.760 (0.763 / 0.753 / 0.711) | 0.572 | 0.733 | 87.2% | -0.096 (p = 0.048) |
| **Overall Pooled** | **13,082** | **3,273** | **0.610** (0.621 / 0.613 / 0.515) | **0.786** (0.788 / 0.782 / 0.741) | **0.568** | **0.731** | **87.1%** | Mean 90% CI Width = 2.130 |

Mean dummy regressor baseline across the identical holdout split (`N = 3,273`): `R2 = 0.000`, `MAE = 0.961`, `RMSE = 1.171`.
<!-- /BENCHMARKS_TABLE -->

## Reproducibility & Pipeline

To run the full pure `pKi` curation, 3-model ensemble training, 5-fold MAPIE conformal calibration, Y-randomization, TreeSHAP, and scaffold-split evaluation pipeline, followed by the automated test suite:

```bash
# 1. Run full curation, training, calibration, and evaluation pipeline
python -m src.run_pipeline

# 2. Run automated tests
pytest
```

## Repository Layout

```text
Adenosine_Receptor_Profiler-main/
├── app.py                  # FastAPI web server and REST API entrypoint
├── web/                    # HTML/CSS/JS frontend application
├── environment.yml         # Conda Python 3.12 environment specification (conda-forge)
├── requirements.txt        # Core runtime Python dependencies
├── pyproject.toml          # Package metadata, optional extras, ruff/mypy/pytest config
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── src/                    # Curation, featurization, training, evaluation, and inference
│   ├── presets.py          # Curated reference ligands and GPCRdb/PDB structure metadata
│   ├── analysis.py         # Nearest-neighbor lookup and local TreeSHAP helpers
│   └── models/             # Ensemble loaders, conformal scaling, ADMET, and SMARTS screens
├── data/
│   ├── raw/                # GPCRdb_*.xlsx and raw ChEMBL source files
│   └── processed/          # A1_pki.csv, A2A_pki.csv, A2B_pki.csv, A3_pki.csv, train_fps.pkl
├── models/                 # Trained XGBoost, LightGBM, Random Forest models and scaler
├── outputs/                # Evaluation JSONs, Y-randomization, SHAP, and diagnostics
└── tests/                  # Pytest unit and integration suite
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
