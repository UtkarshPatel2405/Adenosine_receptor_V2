"""Tab 5: Adenosine Safety Panel, PDE Cross-Reactivity & CNS-MPO / BBB Profiling."""
import pandas as pd
import streamlit as st


def render_tab_safety(data: dict) -> None:
    safe = data.get("safety_profile", {})
    cns = data.get("cns_admet", {})

    st.markdown("""
    <div class="cadd-card">
        <div class="section-num">05</div>
        <div class="section-title" style="color:var(--amber)">Adenosine Safety Panel & CNS-MPO / BBB Profile</div>
        <div class="section-subtitle">Cardiac AV block risk, mast cell degranulation, phosphodiesterase (PDE) cross-reactivity, and blood-brain barrier permeability</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Adenosine Safety Panel (Pillar 3)
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin-bottom:0.4rem'>Adenosine-Specific Target Liabilities:</div>", unsafe_allow_html=True)
    s_cols = st.columns(3)
    with s_cols[0]:
        brady = safe.get("a1_bradycardia_risk", "Low")
        b_col = "var(--red)" if "HIGH" in brady else "var(--amber)" if "MODERATE" in brady else "var(--green)"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">A1 Bradycardia / AV Block</div><div class="kpi-value" style="color:{b_col};font-size:0.9rem">{brady}</div></div>', unsafe_allow_html=True)
    with s_cols[1]:
        mast = safe.get("a3_mast_cell_risk", "Low")
        m_col = "var(--red)" if "HIGH" in mast else "var(--amber)" if "MODERATE" in mast else "var(--green)"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">A3 Mast Cell Degranulation</div><div class="kpi-value" style="color:{m_col};font-size:0.9rem">{mast}</div></div>', unsafe_allow_html=True)
    with s_cols[2]:
        pde = safe.get("pde_cross_reactivity", "Low")
        p_col = "var(--red)" if "HIGH" in pde else "var(--amber)" if "MODERATE" in pde else "var(--green)"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">PDE1-10 Cross-Reactivity</div><div class="kpi-value" style="color:{p_col};font-size:0.9rem">{pde}</div></div>', unsafe_allow_html=True)

    # 2. Pfizer CNS-MPO & Tissue Targeting (Pillar 4)
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.4rem'>Pfizer CNS-MPO & Blood-Brain Barrier (BBB) Permeability:</div>", unsafe_allow_html=True)
    c_cols = st.columns(3)
    with c_cols[0]:
        score_val = cns.get("cns_mpo_score", 0.0)
        cns_col = "var(--green)" if score_val >= 4.0 else "var(--amber)" if score_val >= 3.0 else "var(--cyan)"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Pfizer CNS-MPO Score</div><div class="kpi-value" style="color:{cns_col};font-size:1.15rem">{score_val:.2f} / 6.00</div></div>', unsafe_allow_html=True)
    with c_cols[1]:
        log_bb = cns.get("log_bb", 0.0)
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Predicted LogBB (Brain/Plasma)</div><div class="kpi-value" style="color:var(--purple);font-size:1rem">{log_bb:.2f}</div></div>', unsafe_allow_html=True)
    with c_cols[2]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Tissue Targeting Strategy</div><div class="kpi-value" style="color:var(--cyan);font-size:0.88rem">{cns.get("bbb_status", "N/A")}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="theory-callout"><h4>Translational Tissue Profile</h4>{cns.get("cns_class", "")}</div>', unsafe_allow_html=True)
