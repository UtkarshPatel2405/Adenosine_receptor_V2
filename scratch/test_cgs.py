from src.predictor import predict, _load_xgb_models, _load_scaler
from src.features import build_features
from rdkit import Chem

cgs_smiles = "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O"
print("CGS-21680 Canonical SMILES:", Chem.MolToSmiles(Chem.MolFromSmiles(cgs_smiles)))

res = predict(cgs_smiles)
print("\nPredict Result for CGS-21680:")
print("In database:", res.get("in_database"))
print("DB values:", res.get("db_value"))
print("Predictions:")
for m, sub_dict in res.get("predictions", {}).items():
    print(f"  {m}: {sub_dict}")
print("Intervals:")
for m, iv_dict in res.get("intervals", {}).items():
    print(f"  {m}: {iv_dict}")
print("Selectivity Profile:", res.get("selectivity_profile"))
