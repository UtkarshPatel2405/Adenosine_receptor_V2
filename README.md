# Adenosine Receptor Selectivity & Affinity Profiler

**Rapid in silico pChEMBL profiling across A₁, A₂A, A₂B, A₃ · XGBoost + Random Forest + LightGBM + Stacked ensemble with MAPIE conformal prediction**

| Overall R² | Overall MAE | Compounds | 90% Conformal Coverage | Validation |
| :---: | :---: | :---: | :---: | :---: |
| **0.611** | **0.591** | 18,452 | 85.8% | Bemis–Murcko Scaffold CV |

**Web Application:** [Adenosine Receptor Selectivity Profiler — Live App](https://adenosinereceptorprofiler-htvm52uytmx5hg82pzslwc.streamlit.app/)

---

## 1. Overview

Adenosine receptors (ARs) are class A G protein-coupled receptors (GPCRs) comprising four
human subtypes — A₁, A₂A, A₂B, and A₃ — that mediate cardiorespiratory, neurological,
and oncological signalling. The four subtypes share >70% transmembrane sequence identity,
which makes subtype-selective ligand design a central challenge in GPCR drug discovery.

This platform predicts **pChEMBL binding affinities** (−log₁₀[M]) for small molecules across
all four human adenosine receptor subtypes, quantifies **pairwise subtype selectivity**
(ΔpChEMBL), and reports **distribution-free 90% confidence intervals** via conformal prediction.

## 2. Dataset

- **18,452 unique parent compounds** (14,966 training / 3,486 test) curated from ChEMBL (v34+)
  and cross-referenced with GPCRdb annotations.
- Quality controls: assay confidence ≥ 6, standard relations only (Ki, Kd, IC₅₀, EC₅₀),
  binding measurements prioritised over functional assays, counterion/solvent stripping and
  charge neutralisation via RDKit canonicalisation.
- Verified structural P2Y decoys provide explicit negative controls.

## 3. Methods

1. **Descriptors** — curated 41 physicochemical properties (MolLogP, TPSA, HBD, MW,
   aromatic rings, partial charges) plus Morgan and MACCS fingerprints.
2. **Model ensemble** — XGBoost, Random Forest, and LightGBM base regressors combined in a
   stacked ensemble with a ridge meta-learner.
3. **Conformal prediction** — every regressor is wrapped in a MAPIE
   `CrossConformalRegressor` (Jackknife+), giving finite-sample calibrated intervals
   without distributional assumptions.
4. **Leakage-free evaluation** — global **Bemis–Murcko scaffold split** (random_state 42,
   20% test); entire scaffolds are confined to one partition so no test structure shares a
   ring system with training data.
5. **Selectivity models** — direct pairwise ΔpChEMBL XGBoost regressors for each Aᵢ-vs-Aⱼ
   subtype pair.

## 4. Results — Out-of-Distribution Scaffold Test Set

| Subtype | n_test | R² | MAE | RMSE | 90% Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 3,486 | **0.611** | **0.591** | 0.768 | 85.8% |
| A₁ | 884 | 0.406 | 0.654 | 0.845 | 85.1% |
| A₂A | 1,237 | 0.692 | 0.541 | 0.700 | 88.4% |
| A₂B | 404 | 0.673 | 0.562 | 0.723 | 81.9% |
| A₃ | 961 | 0.599 | 0.610 | 0.795 | 84.7% |

Empirical coverage at the 90% confidence target is 85.8% overall — consistent with the
finite-sample guarantees of Jackknife+ conformal prediction.

## 5. Validation

### 5.1 Y-Randomization (null-model control)

Target labels were shuffled 15× per subtype to break the structure–activity relationship.
Real-model R² collapses to negative shuffled R² in every subtype, confirming the models
learn genuine chemistry rather than dataset artifacts:

| Subtype | Real R² | Shuffled R² (mean ± std) | Leakage |
| :--- | :---: | :---: | :---: |
| A₁ | 0.356 | −0.153 ± 0.061 | No |
| A₂A | 0.617 | −0.100 ± 0.046 | No |
| A₂B | 0.556 | −0.147 ± 0.057 | No |
| A₃ | 0.560 | −0.122 ± 0.047 | No |

### 5.2 External validation (blind literature set)

15 novel literature molecules withheld from training: 15/15 successful predictions, 0 errors,
selectivity recall@1 = **75%**.

### 5.3 Interpretability

TreeSHAP attributions confirm the top drivers are pharmacologically meaningful properties
(MolLogP, TPSA, hydrogen-bond donors, partial charges) and nitrogenous/aromatic fingerprint
motifs, rather than spurious descriptors.

## 6. Repository Layout

```
streamlit_app.py            # Web application (Streamlit entrypoint)
src/                        # Core library (features, predictors, conformal, shap, diagnostics)
models/                     # Production conformal, stacked and selectivity models (Git LFS)
data/                       # Raw ChEMBL/GPCRdb inputs + processed training artifacts
outputs/                    # Evaluation, validation and SHAP reports (JSON/PNG)
scripts/                    # Data acquisition and lookup-build tooling
tests/                      # Pytest suite
```

## 7. Getting Started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Model artifacts are versioned with Git LFS; data artifacts with DVC (`dvc pull`).

## 8. License

MIT — see [LICENSE](LICENSE).