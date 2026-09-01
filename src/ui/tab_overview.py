"""Tab 1: Executive Overview, 4-Subtype Affinity Table & Data Export Center."""
import json
import pandas as pd
import streamlit as st


def render_tab_overview(data: dict) -> None:
    preds = data.get("predictions", {})
    iv = data.get("intervals", {})
    ki_vals = data.get("ki_values", {})
    xgb = preds.get("XGBoost", {})
    best_target = data.get("best_target", "N/A")
    ad_obj = data.get("applicability_domain", {})
    in_db = data.get("in_database", False)
    db_val = data.get("db_value") or {}
    subtypes_list = ["A1", "A2A", "A2B", "A3"]

    sel_spectrum = data.get("selectivity_spectrum", {})
    sel_badge = sel_spectrum.get("classification", "Equipotent Profile")
    sel_color_kpi = "var(--green)" if "Selective" in sel_badge else "var(--cyan)" if "Preferring" in sel_badge else "var(--purple)"

    eff = data.get("functional_efficacy", {})
    moa_lbl = eff.get("mode_of_action", "Unknown")
    moa_col = "var(--green)" if "Agonist" in moa_lbl else "var(--cyan)" if "Antagonist" in moa_lbl else "var(--amber)"

    ad_in = ad_obj.get("in_domain", True)
    ad_status = ad_obj.get("domain_status", "Inside AD" if ad_in else "Outside AD")
    ad_color = "var(--green)" if "Inside" in ad_status else "var(--amber)" if "Borderline" in ad_status else "var(--red)"

    # Execution Mode Banner
    if in_db and any(pd.notna(db_val.get(s)) for s in subtypes_list):
        st.markdown("""
        <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.4);border-radius:8px;padding:0.8rem 1.1rem;margin-bottom:1rem;">
            <div style="font-weight:700;font-size:0.92rem;color:#6ee7b7">✓ MOLECULE IDENTIFIED IN CURATED BIOACTIVITY DATABASE (ChEMBL / GPCRdb)</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.3rem;line-height:1.4">Historical laboratory assay measurements loaded alongside de novo machine learning predictions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.4);border-radius:8px;padding:0.8rem 1.1rem;margin-bottom:1rem;">
            <div style="font-weight:700;font-size:0.92rem;color:#38bdf8">🔬 DE NOVO SCREENING CANDIDATE (NOT FOUND IN TRAINING DATABASE)</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.3rem;line-height:1.4">Full ML pipeline executed across 4 tree ensembles + 7-TM GPCR covariance regularization.</div>
        </div>
        """, unsafe_allow_html=True)

    if not ad_in:
        st.markdown(f'<div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.4);border-radius:8px;padding:0.75rem 1rem;margin-bottom:1rem;font-size:0.8rem;color:#fca5a5;"><strong style="color:#f87171">⚠️ Applicability Domain Advisory:</strong> Low structural overlap (Tanimoto max = {ad_obj.get("tanimoto_max", 0):.2f}). Conformal intervals expanded.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cadd-card">
        <div class="section-num">01</div>
        <div class="section-title" style="color:var(--cyan)">Executive Overview & 4-Subtype Affinity Grid</div>
        <div class="section-subtitle">Multi-model ensemble predictions across A1, A2A, A2B, and A3 with adaptive 90% conformal intervals & thermodynamic Ki</div>
    </div>
    """, unsafe_allow_html=True)

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        top_ki_str = ki_vals.get(best_target, {}).get("display", "N/A")
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Primary Target & Ki</div><div class="kpi-value" style="color:var(--cyan)">{best_target} ({top_ki_str})</div></div>', unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Predicted Mode of Action</div><div class="kpi-value" style="color:{moa_col};font-size:0.95rem">{moa_lbl}</div></div>', unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Selectivity Margin</div><div class="kpi-value" style="color:{sel_color_kpi};font-size:0.92rem">{sel_badge}</div></div>', unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Applicability Domain</div><div class="kpi-value" style="color:{ad_color};font-size:0.92rem">{ad_status}</div></div>', unsafe_allow_html=True)

    # 1. Primary 4-Subtype Profile Table with Clean Columns
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:0.8rem 0 0.3rem'>Primary 4-Subtype Affinity & Conformal Profile:</div>", unsafe_allow_html=True)
    primary_rows = []
    for s in subtypes_list:
        val = xgb.get(s, 0.0)
        ci = iv.get("XGBoost", {}).get(s, {})
        low = ci.get("lower", val - 0.5)
        high = ci.get("upper", val + 0.5)
        ki_str = ki_vals.get(s, {}).get("display", f"{10**(9-val):.1f} nM")
        exp_val = db_val.get(s)
        exp_str = f"{exp_val:.2f} pChEMBL" if (exp_val is not None and pd.notna(exp_val)) else "—"
        primary_rows.append({
            "Receptor Subtype": f"Human {s}",
            "Predicted Affinity (pChEMBL)": f"{val:.2f}",
            "90% Conformal Range": f"[{low:.2f} - {high:.2f}]",
            "Thermodynamic Ki": ki_str,
            "Experimental Ground Truth": exp_str,
        })
    df_primary = pd.DataFrame(primary_rows)
    st.dataframe(df_primary, width="stretch", hide_index=True)

    # 2. Multi-Model Architecture Comparison Table
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:0.8rem 0 0.3rem'>Multi-Model Architecture Consensus (pChEMBL):</div>", unsafe_allow_html=True)
    grid_data = []
    for m in ["XGBoost", "RandomForest", "LightGBM", "Stacked", "MultiTask_Covariance"]:
        row = {"Model Architecture": m}
        for s in subtypes_list:
            v = preds.get(m, {}).get(s, None)
            row[f"Human {s}"] = f"{v:.2f}" if v is not None and isinstance(v, (int, float)) and v > 0 else "—"
        grid_data.append(row)
    st.dataframe(pd.DataFrame(grid_data), width="stretch", hide_index=True)

    # 3. Raw & Processed Data Download Center
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.3rem'>Raw & Processed Data Download Center:</div>", unsafe_allow_html=True)
    d_cols = st.columns(3)
    with d_cols[0]:
        csv_data = df_primary.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary (CSV)", data=csv_data, file_name="adenosine_profile.csv", mime="text/csv", use_container_width=True)
    with d_cols[1]:
        # Filter non-serializable fields for JSON export
        clean_json = {k: v for k, v in data.items() if not k.startswith("mol_block_") and not k.startswith("svg_")}
        json_data = json.dumps(clean_json, indent=2, default=str).encode('utf-8')
        st.download_button("📥 Download Payload (JSON)", data=json_data, file_name="adenosine_payload.json", mime="application/json", use_container_width=True)
    with d_cols[2]:
        sdf_data = data.get("mol_block_3d", "")
        if sdf_data:
            st.download_button("📥 Download 3D Conformer (SDF)", data=sdf_data, file_name="conformer_3d.sdf", mime="chemical/x-mdl-sdfile", use_container_width=True)
