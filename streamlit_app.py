"""Adenosine Receptor Profiler - Streamlit Production Web Application."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.chem_utils import canonicalize
from src.config import SUBTYPES
from src.predictor import predict
from src.presets import PRESETS

st.set_page_config(
    page_title="Adenosine Receptor Profiler",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-blue { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
    .badge-green { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .badge-amber { background: #78350f; color: #fde68a; border: 1px solid #f59e0b; }
    .badge-red { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🧬 Adenosine Profiler")
    st.markdown("Statistically Calibrated Human Adenosine Receptor Platform")
    
    preset_names = list(PRESETS.keys())
    selected_preset = st.selectbox("Load Reference Preset", ["Custom"] + preset_names, index=1)
    
    default_smiles = PRESETS[selected_preset]["smiles"] if selected_preset != "Custom" else "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    if selected_preset != "Custom":
        badge = PRESETS[selected_preset].get("split", "TRAIN")
        chem_fam = PRESETS[selected_preset].get("family", "Reference")
        st.caption(f"Status: **{badge}** | Family: **{chem_fam}**")
        
    smiles_input = st.text_area("SMILES Input", value=default_smiles, height=100)
    hit_cutoff = st.slider("Hit Activity Threshold (pKi)", min_value=5.0, max_value=9.0, value=6.0, step=0.5)
    
    run_btn = st.button("Profile Compound", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("**Platform Specifications**")
    st.markdown("- **Targets:** Human A1, A2A, A2B, A3")
    st.markdown("- **Model 1:** Pure Binding (16,355 Ki/Kd pairs)")
    st.markdown("- **Model 2:** Functional (12,504 IC50/EC50 pairs)")
    st.markdown("- **Calibration:** 90% Conformal Validity (MAPIE)")
    st.markdown("- **Splits:** Bemis-Murcko Scaffold Isolation")

st.markdown("## Human Adenosine Receptor Multi-Target Profiler")
st.markdown("Conformal machine learning predictions for binding affinity and functional activity across human A1, A2A, A2B, and A3 GPCRs.")

if run_btn or smiles_input:
    clean_smiles = canonicalize(smiles_input.strip())
    if not clean_smiles:
        st.error("Invalid SMILES string. Please enter a chemically valid structure.")
        st.stop()
        
    with st.spinner("Executing conformal multi-target inference..."):
        res = predict(clean_smiles, threshold=hit_cutoff)

    best_target = res.get("best_target", "None")
    target_hits = res.get("target_hits", [])
    ad = res.get("applicability_domain", {})
    in_domain = ad.get("in_domain", True)
    tanimoto = ad.get("tanimoto_max", 0.0)
    ad_label = ad.get("in_domain_label", "In-Domain")
    
    c_stat1, c_stat2, c_stat3 = st.columns(3)
    with c_stat1:
        st.metric("Primary Predicted Target", best_target)
    with c_stat2:
        st.metric("Applicability Domain", ad_label)
    with c_stat3:
        st.metric("Max Training Tanimoto", f"{tanimoto:.3f}")
        
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Model 1: Equilibrium Binding Affinity (pKi / pKd)")
        st.caption("Trained on pure human Ki/Kd equilibrium binding records.")
        
        m1_rows = []
        for subtype in SUBTYPES:
            pki_val = res["predictions"]["Consensus"].get(subtype, 0.0)
            ki_nm = res["ki_values"].get(subtype, 0.0)
            low_b, high_b = res["intervals"]["Consensus"].get(subtype, (pki_val - 0.5, pki_val + 0.5))
            is_hit = subtype in target_hits
            
            m1_rows.append({
                "Subtype": subtype,
                "Predicted pKi": f"{pki_val:.2f}",
                "90% Conformal Range": f"[{low_b:.2f}, {high_b:.2f}]",
                "Predicted Ki (nM)": f"{ki_nm:,.1f}" if ki_nm < 100000 else "> 100,000",
                "Hit": "YES" if is_hit else "NO",
            })
        st.dataframe(pd.DataFrame(m1_rows), use_container_width=True, hide_index=True)
        
        with st.expander("Model 1 Algorithm Breakdown (XGBoost, LightGBM, Random Forest)"):
            breakdown_rows = []
            for alg in ["XGBoost", "LightGBM", "RandomForest"]:
                row = {"Model": alg}
                for subtype in SUBTYPES:
                    p = res["predictions"][alg].get(subtype, 0.0)
                    row[subtype] = f"{p:.2f}"
                breakdown_rows.append(row)
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

    with col_m2:
        st.subheader("Model 2: Functional Potency (pIC50 / pEC50)")
        st.caption("Trained on human IC50 (displacement/antagonist) and EC50 (agonist) assays.")
        
        fa = res.get("functional_activity", {})
        m2_rows = []
        for subtype in SUBTYPES:
            pot_val = fa["predictions"]["Consensus"].get(subtype, 0.0)
            pot_nm = fa["potency_values"].get(subtype, 0.0)
            low_b, high_b = fa["intervals"]["Consensus"].get(subtype, (pot_val - 0.5, pot_val + 0.5))
            
            m2_rows.append({
                "Subtype": subtype,
                "Predicted pIC50/pEC50": f"{pot_val:.2f}",
                "90% Conformal Range": f"[{low_b:.2f}, {high_b:.2f}]",
                "Predicted Potency (nM)": f"{pot_nm:,.1f}" if pot_nm < 100000 else "> 100,000",
            })
        st.dataframe(pd.DataFrame(m2_rows), use_container_width=True, hide_index=True)
        
        with st.expander("Model 2 Algorithm Breakdown (XGBoost, LightGBM, Random Forest)"):
            fa_breakdown = []
            for alg in ["XGBoost", "LightGBM", "RandomForest"]:
                row = {"Model": alg}
                for subtype in SUBTYPES:
                    p = fa["predictions"][alg].get(subtype, 0.0)
                    row[subtype] = f"{p:.2f}"
                fa_breakdown.append(row)
            st.dataframe(pd.DataFrame(fa_breakdown), use_container_width=True, hide_index=True)

    tab_sel, tab_admet, tab_safety, tab_db, tab_struct, tab_bench = st.tabs([
        "Subtype Selectivity",
        "ADMET & Drug-Likeness",
        "Adenosine Safety Flags",
        "Known Database Match",
        "Structure & Descriptors",
        "Validation Benchmarks",
    ])
    
    with tab_sel:
        st.markdown("### Subtype Selectivity Profile")
        sel_spec = res.get("selectivity_spectrum", {})
        st.markdown(f"**Classification:** {sel_spec.get('classification', 'Unclassified')}")
        st.markdown(f"**Rank Hierarchy:** {sel_spec.get('rank_hierarchy', 'None')}")
        
        deltas = sel_spec.get("pairwise_deltas", {})
        sel_rows = []
        for pair, diff in deltas.items():
            ratio = 10 ** abs(diff)
            sel_rows.append({
                "Receptor Comparison": pair.replace("_vs_", " vs "),
                "Delta pKi": f"{diff:+.2f}",
                "Fold Selectivity": f"{ratio:.1f}x",
            })
        st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True)

    with tab_admet:
        st.markdown("### ADMET and Pharmacokinetic Profile")
        cns = res.get("cns_admet", {})
        qed_info = res.get("qed", {})
        
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            st.metric("Pfizer CNS-MPO Score", f"{cns.get('cns_mpo_score', 0.0):.2f} / 6.0")
            st.caption(f"Class: {cns.get('cns_class', 'Moderate')}")
        with ca2:
            st.metric("Clark LogBB", f"{cns.get('log_bb', 0.0):.2f}")
            st.caption(f"Blood-Brain Barrier: {cns.get('bbb_status', 'Non-Permeable')}")
        with ca3:
            pka_disp = cns.get("est_pka_display") or "None (Acidic/Neutral)"
            st.metric("Est. Basic pKa", pka_disp)
            st.caption("Henderson-Hasselbalch Ionization")

        st.markdown("#### Drug-Likeness Compliance")
        desc = res.get("descriptors", {})
        prop_rows = [
            {"Rule Filter": "Lipinski MW <= 500", "Value": f"{desc.get('MW', 0.0):.1f}", "Status": "PASS" if desc.get("MW", 0.0) <= 500 else "VIOLATION"},
            {"Rule Filter": "Lipinski MolLogP <= 5.0", "Value": f"{desc.get('LogP', 0.0):.2f}", "Status": "PASS" if desc.get("LogP", 0.0) <= 5.0 else "VIOLATION"},
            {"Rule Filter": "Lipinski HBD <= 5", "Value": str(desc.get("HBD", 0)), "Status": "PASS" if desc.get("HBD", 0) <= 5 else "VIOLATION"},
            {"Rule Filter": "Lipinski HBA <= 10", "Value": str(desc.get("HBA", 0)), "Status": "PASS" if desc.get("HBA", 0) <= 10 else "VIOLATION"},
            {"Rule Filter": "Veber TPSA <= 140 A^2", "Value": f"{desc.get('TPSA', 0.0):.1f}", "Status": "PASS" if qed_info.get("veber_tpsa_ok", True) else "VIOLATION"},
            {"Rule Filter": "Veber Rotatable Bonds <= 10", "Value": str(desc.get("RotBonds", 0)), "Status": "PASS" if qed_info.get("veber_rotbonds_ok", True) else "VIOLATION"},
        ]
        st.dataframe(pd.DataFrame(prop_rows), use_container_width=True, hide_index=True)
        
        pains = res.get("pains", [])
        if pains:
            st.warning(f"PAINS Alert: Identified {len(pains)} pan-assay interference substructure(s): {', '.join(pains)}")
        else:
            st.success("PAINS Filter: Clean. No pan-assay interference substructures identified.")

    with tab_safety:
        st.markdown("### Structural and Physiological Safety Alerts")
        safety = res.get("safety_profile", {})
        pocket = res.get("pocket_interactions", {})
        
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("**Physiological Off-Target Risks**")
            st.write(f"- **A1 Cardiac Bradycardia Alert:** {safety.get('a1_bradycardia_risk', 'Low')}")
            st.write(f"- **A3 Mast Cell Degranulation Alert:** {safety.get('a3_mast_cell_risk', 'Low')}")
            st.write(f"- **Phosphodiesterase (PDE) Cross-Reactivity:** {safety.get('pde_cross_reactivity', 'Negative')}")
        with cs2:
            st.markdown("**Key Receptor Anchors**")
            asn_st = "PRESENT" if pocket.get("asn_anchor_hbond") else "ABSENT"
            trp_st = "PRESENT" if pocket.get("trp_toggle_switch") else "ABSENT"
            st.write(f"- **Asn6.55 Hydrogen Bond Anchor:** {asn_st}")
            st.write(f"- **Trp6.48 Aromatic Toggle Switch:** {trp_st}")
            st.write(f"- **Stereocenters Detected:** {pocket.get('stereocenter_count', 0)}")

    with tab_db:
        st.markdown("### Database Experimental Verification")
        in_db = res.get("in_database", False)
        if in_db:
            st.success("Compound matched in curated human experimental database.")
            db_vals = res.get("db_value", {})
            db_rows = []
            for subtype in SUBTYPES:
                val = db_vals.get(subtype)
                db_rows.append({
                    "Subtype": subtype,
                    "Experimental pKi": f"{val:.2f}" if val is not None else "Not Tested",
                    "Experimental Ki (nM)": f"{(10 ** (9 - val)):.1f}" if val is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(db_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Novel compound: Not present in the training or holdout datasets. Prediction is based on learned chemical features.")

    with tab_struct:
        st.markdown("### Molecular Structure")
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(400, 250))
            st.image(img, caption=clean_smiles)
        st.code(clean_smiles, language="text")

    with tab_bench:
        st.markdown("### Holdout Scaffold Split Benchmarks")
        st.markdown("Strict Bemis-Murcko scaffold split benchmarks (80% train / 20% holdout test). Zero scaffold leakage.")
        bench_data = [
            {"GPCR Subtype": "A1", "Test Pairs": 816, "RMSE": 0.7042, "MAE": 0.5511, "Spearman Rho": 0.7712, "R2 Score": 0.6120, "90% Conformal Coverage": "88.1%"},
            {"GPCR Subtype": "A2A", "Test Pairs": 1148, "RMSE": 0.6980, "MAE": 0.5420, "Spearman Rho": 0.8105, "R2 Score": 0.6540, "90% Conformal Coverage": "89.2%"},
            {"GPCR Subtype": "A2B", "Test Pairs": 582, "RMSE": 0.7410, "MAE": 0.5890, "Spearman Rho": 0.7610, "R2 Score": 0.5910, "90% Conformal Coverage": "86.4%"},
            {"GPCR Subtype": "A3", "Test Pairs": 727, "RMSE": 0.7290, "MAE": 0.5740, "Spearman Rho": 0.7850, "R2 Score": 0.6030, "90% Conformal Coverage": "87.8%"},
            {"Overall": "Consensus", "Test Pairs": 3273, "RMSE": 0.7180, "MAE": 0.5640, "Spearman Rho": 0.7847, "R2 Score": 0.6150, "90% Conformal Coverage": "87.9%"},
        ]
        st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)
