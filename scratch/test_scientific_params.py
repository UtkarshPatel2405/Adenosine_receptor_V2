from src.predictor import predict
from src.applicability_domain import check_applicability_domain
from src.models.selectivity_engine import compute_selectivity_spectrum

test_molecules = {
    "Aspirin (Out of Domain Non-binder)": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol (Inactive Non-binder)": "CC(=O)Nc1ccc(O)cc1",
    "CGS-21680 (A2A Canonical Potent Agonist)": "CCNC(=O)C1OC(n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)C(O)C1O",
    "PSB-603 (A2B Selective Antagonist)": "CCCn1c(=O)c2[nH]c(-c3ccc(cc3)S(=O)(=O)N3CCN(c4ccc(Cl)cc4)CC3)nc2c(=O)n1CCC",
    "CCPA (A1 Potent Agonist)": "Clc1nc(NC2CCCC2)c2ncn(C3OC(CO)C(O)C3O)c2n1",
}

print("=== Scientific Parameter & Calibration Validation ===")
for name, smi in test_molecules.items():
    res = predict(smi)
    ad = check_applicability_domain(smi)
    spec = res["selectivity_spectrum"]
    xgb = res["predictions"]["XGBoost"]
    print(f"\n[{name}]")
    print(f"  • In Database: {res['in_database']}")
    print(f"  • Applicability Domain: {ad['domain_status']} (Tanimoto Max = {ad['tanimoto_max']})")
    print(f"  • Predicted pChEMBL: A1={xgb['A1']:.2f}, A2A={xgb['A2A']:.2f}, A2B={xgb['A2B']:.2f}, A3={xgb['A3']:.2f}")
    print(f"  • Scientific Selectivity Class: {spec['classification']}")
    print(f"  • Margin: {spec['selectivity_margin']:.2f} delta ({spec['selectivity_fold']:.1f}x)")
