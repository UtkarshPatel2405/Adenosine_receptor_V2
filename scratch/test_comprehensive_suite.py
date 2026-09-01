from src.predictor import predict

test_suite = {
    "CGS-21680 (A2A Selective Agonist)": "CCNC(=O)C1OC(n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)C(O)C1O",
    "Regadenoson (FDA Approved A2A Agonist)": "CNC(=O)c1cnn(c1)-c1nc(N)c2ncn(C3OC(CO)C(O)C3O)c2n1",
    "Istradefylline (FDA Approved A2A Antagonist)": "CCN1C(=O)C2=C(N=C(N2C)/C=C/c3ccc(OC)c(OC)c3)N(C1=O)CC",
    "PSB-603 (A2B Selective Antagonist)": "CCCn1c(=O)c2[nH]c(-c3ccc(cc3)S(=O)(=O)N3CCN(c4ccc(Cl)cc4)CC3)nc2c(=O)n1CCC",
    "CCPA (A1 Selective Agonist)": "Clc1nc(NC2CCCC2)c2ncn(C3OC(CO)C(O)C3O)c2n1",
    "IB-MECA (A3 Selective Agonist)": "CNC(=O)C1OC(n2cnc3c(NCc4ccccc4)ncnc32)C(O)C1O",
    "Aspirin (Out-of-Domain Non-binder)": "CC(=O)Oc1ccccc1C(=O)O",
}

print("=" * 80)
print("COMPREHENSIVE INDUSTRIAL CADD BENCHMARK EVALUATION")
print("=" * 80)

for name, smi in test_suite.items():
    res = predict(smi)
    eff = res["functional_efficacy"]
    pkt = res["pocket_interactions"]
    safe = res["safety_profile"]
    cns = res["cns_admet"]
    ad = res["applicability_domain"]
    ki = res["ki_values"]
    xgb = res["predictions"]["XGBoost"]
    target = res["best_target"]

    print(f"\n[{name}]")
    print(f"  • Primary Target: {target} (pChEMBL = {xgb.get(target, 0):.2f}, Ki = {ki.get(target, {}).get('display', '')})")
    print(f"  • Mode of Action: {eff.get('mode_of_action')} (Act Prob: {eff.get('activation_probability', 0)*100:.1f}%)")
    print(f"  • Downstream Signaling: {eff.get('signaling_pathway')}")
    print(f"  • Asn6.55 Anchor: {'Formed' if pkt.get('asn_anchor_hbond') else 'Missing'} | Trp6.48 Toggle: {'Active' if pkt.get('trp_toggle_switch') else 'Inactive'}")
    print(f"  • Safety: A1 Bradycardia = {safe.get('a1_bradycardia_risk')} | PDE Liability = {safe.get('pde_cross_reactivity')}")
    print(f"  • CNS Targeting: MPO = {cns.get('cns_mpo_score')} / 6.0 | LogBB = {cns.get('log_bb')} | Status = {cns.get('bbb_status')}")
    print(f"  • Applicability Domain: {ad.get('domain_status')} (Tanimoto Max = {ad.get('tanimoto_max')})")

print("\n" + "=" * 80)
