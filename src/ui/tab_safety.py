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
    st.markdown("<div style='font-size:0.88rem;font-weight:700;color:#f8fafc;margin-bottom:0.5rem'>Adenosine-Specific Target Liabilities:</div>", unsafe_allow_html=True)
    s_cols = st.columns(3)
    
    with s_cols[0]:
        brady = safe.get("a1_bradycardia_risk", "Low Risk")
        b_col = "var(--red)" if "HIGH" in brady.upper() else "var(--amber)" if "MODERATE" in brady.upper() else "var(--green)"
        b_badge = "badge-red" if "HIGH" in brady.upper() else "badge-amber" if "MODERATE" in brady.upper() else "badge-green"
        st.markdown(f"""
        <div class="target-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <span style="font-weight:700;color:#f8fafc">A1 Cardiac Liability</span>
                <span class="badge-pill {b_badge}">{brady}</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted)">Bradycardia & AV Nodal Blockade</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.35rem">Threshold: pChEMBL(A1) ≥ 7.0</div>
        </div>
        """, unsafe_allow_html=True)

    with s_cols[1]:
        mast = safe.get("a3_mast_cell_risk", "Low Risk")
        m_col = "var(--red)" if "HIGH" in mast.upper() else "var(--amber)" if "MODERATE" in mast.upper() else "var(--green)"
        m_badge = "badge-red" if "HIGH" in mast.upper() else "badge-amber" if "MODERATE" in mast.upper() else "badge-green"
        st.markdown(f"""
        <div class="target-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <span style="font-weight:700;color:#f8fafc">A3 Mast Cell Liability</span>
                <span class="badge-pill {m_badge}">{mast}</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted)">Degranulation & Bronchoconstriction</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.35rem">Threshold: pChEMBL(A3) ≥ 7.0</div>
        </div>
        """, unsafe_allow_html=True)

    with s_cols[2]:
        pde = safe.get("pde_cross_reactivity", "Low Risk")
        p_col = "var(--red)" if "HIGH" in pde.upper() else "var(--amber)" if "MODERATE" in pde.upper() else "var(--green)"
        p_badge = "badge-red" if "HIGH" in pde.upper() else "badge-amber" if "MODERATE" in pde.upper() else "badge-green"
        st.markdown(f"""
        <div class="target-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <span style="font-weight:700;color:#f8fafc">PDE Cross-Reactivity</span>
                <span class="badge-pill {p_badge}">{pde}</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted)">Xanthine Off-Target Inhibition</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.35rem">Purine / Heterocycle screening</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Pfizer CNS-MPO & Tissue Targeting (Pillar 4)
    st.markdown("<div style='font-size:0.88rem;font-weight:700;color:#f8fafc;margin:1.3rem 0 0.5rem'>Pfizer CNS-MPO & Blood-Brain Barrier (BBB) Permeability:</div>", unsafe_allow_html=True)
    c_cols = st.columns(3)
    score_val = float(cns.get("cns_mpo_score", 0.0) or 0.0)
    score_pct = max(0.0, min(100.0, score_val / 6.0 * 100.0))
    cns_col = "var(--green)" if score_val >= 4.0 else "var(--amber)" if score_val >= 3.0 else "var(--cyan)"
    
    with c_cols[0]:
        st.markdown(f"""
        <div class="target-card">
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">Pfizer CNS-MPO Score</div>
            <div style="display:flex;align-items:baseline;gap:0.4rem;margin:0.2rem 0">
                <span style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;color:{cns_col}">{score_val:.2f}</span>
                <span style="font-size:0.85rem;color:var(--text-muted)">/ 6.00</span>
            </div>
            <div class="potency-bar-track">
                <div class="potency-bar-fill" style="width:{score_pct}%;background:linear-gradient(90deg, {cns_col}88, {cns_col})"></div>
            </div>
            <div style="font-size:0.73rem;color:var(--text-muted);margin-top:0.25rem">≥ 4.0 = High CNS Permeability</div>
        </div>
        """, unsafe_allow_html=True)

    with c_cols[1]:
        log_bb = float(cns.get("log_bb", 0.0) or 0.0)
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Predicted LogBB (Brain/Plasma)</div>
            <div class="kpi-value" style="color:var(--purple);font-size:1.4rem">{log_bb:.2f}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem">> 0.3 = Rapid Brain Uptake</div>
        </div>
        """, unsafe_allow_html=True)

    with c_cols[2]:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Tissue Targeting Horizon</div>
            <div class="kpi-value" style="color:var(--cyan);font-size:1.05rem">{cns.get("bbb_status", "Peripheral Restricted")}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem">Blood-Brain Barrier Disposition</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="theory-callout">
        <h4>Translational Pharmacology & Safety Classification:</h4>
        {cns.get("cns_class", "Multiparametric optimization profile calculated across CLogP, CLogD7.4, MW, TPSA, HBD, and pKa.")}
    </div>
    """, unsafe_allow_html=True)

