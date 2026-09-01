"""Tab 4: QED Drug-Likeness Profile & PAINS Substructure Liability Screen."""
import streamlit as st


def render_tab_druglikeness(data: dict) -> None:
    qed_data = data.get("qed_profile", {})
    qed_score = qed_data.get("QED", qed_data.get("qed", 0.5)) or 0.5

    st.markdown("""
    <div class="cadd-card">
        <div class="section-num">06</div>
        <div class="section-title" style="color:var(--amber)">QED & PAINS Drug-Likeness Filter</div>
        <div class="section-subtitle">Quantitative Estimation of Drug-likeness and substructure liability screening</div>
    </div>
    """, unsafe_allow_html=True)

    col_qed, col_lipinski, col_pains = st.columns(3)
    with col_qed:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">QED Score</div>
            <div class="kpi-value" style="color:var(--amber)">{qed_score:.3f}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem">0.0 (Unfavorable) to 1.0 (Drug-like)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_lipinski:
        desc = data.get("descriptors", {})
        mw = desc.get("MW", 0)
        logp = desc.get("LogP", 0)
        hbd = desc.get("HBD", 0)
        hba = desc.get("HBA", 0)
        ro5_pass = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
        ro5_lbl = "Compliant (0 Violations)" if ro5_pass else "Violations Detected"
        ro5_col = "var(--green)" if ro5_pass else "var(--red)"

        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Lipinski Rule of 5</div>
            <div class="kpi-value" style="color:{ro5_col};font-size:0.95rem">{ro5_lbl}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem">MW: {mw:.1f} | LogP: {logp:.2f} | HBD: {hbd} | HBA: {hba}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_pains:
        pains_list = data.get("pains_alerts", []) or []
        p_clean = len(pains_list) == 0
        p_lbl = "Zero PAINS Alerts" if p_clean else f"{len(pains_list)} PAINS Alert(s)"
        p_col = "var(--green)" if p_clean else "var(--red)"

        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">PAINS Filter</div>
            <div class="kpi-value" style="color:{p_col};font-size:0.95rem">{p_lbl}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem">Pan-Assay Interference liability scan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-callout">
        <h4>Theory: Drug-Likeness & Substructure Liabilities</h4>
        QED models the underlying distribution of 8 key physicochemical properties of approved oral drugs using desirability functions. The PAINS filter screens for frequent-hitter motifs that cause assay interference via covalent reactivity, redox cycling, or membrane disruption.
    </div>
    """, unsafe_allow_html=True)
