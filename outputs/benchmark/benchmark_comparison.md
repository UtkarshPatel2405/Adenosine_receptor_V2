# Literature Benchmark Comparison

| Model | Split | Metric | A1 | A2A | A2B | A3 |
|-------|-------|--------|-----|------|------|-----|
| our_model_xgboost | Scaffold (Bemis-Murcko) | R² | 0.753 | 0.884 | 0.912 | 0.886 |
| | | MAE | 0.425 | 0.311 | 0.253 | 0.255 |
| our_model_gnn | Scaffold (Bemis-Murcko) | R² | 0.261 | 0.208 | 0.301 | 0.373 |
| | | MAE | 0.736 | 0.866 | 0.626 | 0.733 |
| Rodríguez-Pérez_2020 | Scaffold | R² | 0.520 | 0.610 | 0.480 | 0.550 |
| | | MAE | 0.580 | 0.510 | 0.550 | 0.540 |
| Salmaso_2022 | Temporal | R² | 0.600 | 0.720 | 0.550 | 0.680 |
| | | MAE | N/A | N/A | N/A | N/A |
| ChEMBL_RF_Baseline | Random | R² | 0.750 | 0.800 | 0.700 | 0.780 |
| | | MAE | 0.420 | 0.380 | 0.450 | 0.400 |