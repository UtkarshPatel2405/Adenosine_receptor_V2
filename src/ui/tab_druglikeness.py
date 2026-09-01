"""Tab 6: QED Drug-Likeness Profile & PAINS Substructure Liability Screen."""
import pandas as pd
import streamlit as st


def render_tab_druglikeness(data: dict) -> None:
    qed_data = data.get("qed_profile", {})
    qed_score = float(qed_data.get("QED", qed_data.get("qed", 0.5)) or 0.5)
    qed_pct = max(0.0, min(100.0, qed_score * 100.0))
    qed_col = "var(--green)" if qed_score >= 0.67 else "var(--amber)" if qed_score >= 0.45 else "var(--red)"

    desc = data.get("descriptors", {})
    mw = float(desc.get("MW", 0) or 0)
    logp = float(desc.get("LogP", 0) or 0)
    hbd = int(desc.get("HBD", 0) or 0)
    hba = int(desc.get("HBA", 0) or 0)
    tpsa = float(desc.get("TPSA", 0) or 0)
    rotb = int(desc.get("RotBonds", 0) or 0)
    
    ro5_violations = sum([mw > 500, logp > 5.0, hbd > 5, hba > 10])
    ro5_pass = ro5_violations == 0
    ro5_lbl = "Lipinski Compliant (0 Violations)" if ro5_pass else f"{ro5_violations} Ro5 Violation(s)"
    ro5_badge = "badge-green" if ro5_pass else "badge-red"

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
        <div class="target-card">
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">Bickerton QED Score</div>
            <div style="display:flex;align-items:baseline;gap:0.4rem;margin:0.2rem 0">
                <span style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;color:{qed_col}">{qed_score:.3f}</span>
                <span style="font-size:0.85rem;color:var(--text-muted)">/ 1.000</span>
            </div>
            <div class="potency-bar-track">
                <div class="potency-bar-fill" style="width:{qed_pct}%;background:linear-gradient(90deg, {qed_col}88, {qed_col})"></div>
            </div>
            <div style="font-size:0.73rem;color:var(--text-muted);margin-top:0.25rem">≥ 0.67 = High Oral Drug-Likeness</div>
        </div>
        """, unsafe_allow_html=True)

    with col_lipinski:
        st.markdown(f"""
        <div class="target-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <span style="font-weight:700;color:#f8fafc">Lipinski Rule of 5</span>
                <span class="badge-pill {ro5_badge}">{ro5_lbl}</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted)">MW ≤ 500 · LogP ≤ 5 · HBD ≤ 5 · HBA ≤ 10</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.35rem">Oral Bioavailability Probability: <b>{'High (>85%)' if ro5_pass else 'Moderate / Reduced'}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_pains:
        pains_list = data.get("pains_alerts", []) or []
        p_clean = len(pains_list) == 0
        p_lbl = "Zero PAINS Alerts" if p_clean else f"{len(pains_list)} PAINS Motifs"
        p_badge = "badge-green" if p_clean else "badge-red"
        st.markdown(f"""
        <div class="target-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <span style="font-weight:700;color:#f8fafc">PAINS Filter</span>
                <span class="badge-pill {p_badge}">{p_lbl}</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted)">Pan-Assay Interference Screening</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:0.35rem">{'No promiscuous covalent / redox alerts' if p_clean else f'Alerts: {", ".join(pains_list)}'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.88rem;font-weight:700;color:#f8fafc;margin:1.3rem 0 0.5rem'>Physicochemical Property Manifold (ADMET Descriptors):</div>", unsafe_allow_html=True)
    p_cols = st.columns(6)
    props = [
        ("Mol Weight", f"{mw:.1f} Da", "MW ≤ 500", mw <= 500),
        ("Lipophilicity", f"{logp:.2f}", "LogP ≤ 5.0", logp <= 5.0),
        ("Polar Surface", f"{tpsa:.1f} Å²", "TPSA ≤ 140", tpsa <= 140),
        ("H-Bond Donors", f"{hbd}", "HBD ≤ 5", hbd <= 5),
        ("H-Bond Acceptors", f"{hba}", "HBA ≤ 10", hba <= 10),
        ("Rotatable Bonds", f"{rotb}", "RotB ≤ 10", rotb <= 10),
    ]
    for idx, (label, val_str, rule, passed) in enumerate(props):
        p_col = "var(--green)" if passed else "var(--amber)"
        with p_cols[idx]:
            st.markdown(f"""
            <div class="kpi-box" style="padding:0.75rem 0.5rem">
                <div class="kpi-label" style="font-size:0.68rem">{label}</div>
                <div class="kpi-value" style="font-size:1.15rem;color:{p_col}">{val_str}</div>
                <div style="font-size:0.68rem;color:var(--text-muted);margin-top:0.2rem">{rule}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-callout">
        <h4>Theory: Drug-Likeness & Substructure Liabilities</h4>
        QED models the underlying distribution of 8 key physicochemical properties of approved oral drugs using desirability functions. The PAINS filter screens for frequent-hitter motifs that cause assay interference via covalent reactivity, redox cycling, or membrane disruption.
    </div>
    """, unsafe_allow_html=True)

