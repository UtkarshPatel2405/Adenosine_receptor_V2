"""Tab 4: Functional Efficacy, Mode of Action (MoA) & 3D Toggle Switch Analysis."""
import pandas as pd
import streamlit as st


def render_tab_efficacy(data: dict) -> None:
    eff = data.get("functional_efficacy", {})
    pkt = data.get("pocket_interactions", {})
    best_target = data.get("best_target", "N/A")

    st.markdown("""
    <div class="cadd-card">
        <div class="section-num">04</div>
        <div class="section-title" style="color:var(--purple)">Functional Efficacy & 3D Activation Toggle Switches</div>
        <div class="section-subtitle">Pharmacological Mode of Action (Agonist vs Antagonist), G-protein cascades, and orthosteric toggle switches</div>
    </div>
    """, unsafe_allow_html=True)

    moa_title = eff.get("mode_of_action", "Unknown")
    moa_color = "var(--green)" if "Agonist" in moa_title else "var(--cyan)" if "Antagonist" in moa_title else "var(--amber)"
    act_prob = float(eff.get("activation_probability", 0.0) or 0.0)
    act_pct = max(0.0, min(100.0, act_prob * 100.0))

    # Top Efficacy & Signaling KPI Strip
    col_moa, col_cascade = st.columns([1.2, 1.8])
    with col_moa:
        st.markdown(f"""
        <div class="target-card primary-hit" style="animation:none">
            <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">Pharmacological Mode of Action</div>
            <div style="font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:800;color:{moa_color};margin:0.25rem 0">{moa_title}</div>
            <div style="font-size:0.78rem;color:#cbd5e1;display:flex;justify-content:space-between">
                <span>Activation Probability:</span>
                <span style="font-weight:700;color:{moa_color}">{act_prob*100:.1f}%</span>
            </div>
            <div class="potency-bar-track" style="margin-top:0.4rem">
                <div class="potency-bar-fill" style="width:{act_pct}%;background:linear-gradient(90deg, {moa_color}88, {moa_color})"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_cascade:
        st.markdown(f"""
        <div class="cadd-card" style="margin-bottom:0">
            <div style="font-size:0.85rem;font-weight:700;color:var(--cyan);margin-bottom:0.3rem">Downstream G-Protein Signaling Pathway:</div>
            <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.5">{eff.get("signaling_pathway", "N/A")}</div>
            <div style="margin-top:0.5rem;font-size:0.82rem;color:#94a3b8">Therapeutic Horizon: <b style="color:#f8fafc">{eff.get("therapeutic_indication", "Cardiovascular / Oncology / Neuroinflammation")}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.88rem;font-weight:700;color:#f8fafc;margin:1.2rem 0 0.5rem'>3D Orthosteric Pocket Anchors & Activation Toggle Switches:</div>", unsafe_allow_html=True)
    i_cols = st.columns(3)
    with i_cols[0]:
        asn_ok = pkt.get("asn_anchor_hbond", False)
        asn_col = "var(--green)" if asn_ok else "var(--red)"
        asn_lbl = "Formed (Dual H-Bond)" if asn_ok else "Missing Anchor"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">{pkt.get("asn_residue", "Asn6.55 (TM6 Primary Anchor)")}</div>
            <div class="kpi-value" style="color:{asn_col};font-size:1.05rem">{asn_lbl}</div>
            <div style="font-size:0.73rem;color:var(--text-muted);margin-top:0.3rem">Purine / Adenine recognition</div>
        </div>
        """, unsafe_allow_html=True)

    with i_cols[1]:
        trp_ok = pkt.get("trp_toggle_switch", False)
        trp_col = "var(--green)" if trp_ok else "var(--amber)"
        trp_lbl = "Active Engagement" if trp_ok else "Inactive Rotamer"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">{pkt.get("trp_residue", "Trp6.48 (Transmission Switch)")}</div>
            <div class="kpi-value" style="color:{trp_col};font-size:1.05rem">{trp_lbl}</div>
            <div style="font-size:0.73rem;color:var(--text-muted);margin-top:0.3rem">CWxP helix-bending toggle</div>
        </div>
        """, unsafe_allow_html=True)

    with i_cols[2]:
        phe_ok = pkt.get("phe_pi_stacking", True)
        phe_col = "var(--green)" if phe_ok else "var(--amber)"
        phe_lbl = "Optimal Pi-Stacking" if phe_ok else "Partial Contact"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Phe168 / Phe171 (ECL2 Lid)</div>
            <div class="kpi-value" style="color:{phe_col};font-size:1.05rem">{phe_lbl}</div>
            <div style="font-size:0.73rem;color:var(--text-muted);margin-top:0.3rem">Hydrophobic pocket gating</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-callout">
        <h4>Molecular Pharmacology of GPCR Activation:</h4>
        Adenosine receptor functional efficacy is governed by engagement with the conserved <b>Asn6.55</b> anchor and steric displacement of the <b>Trp6.48</b> microswitch. Agonists trigger outward movement of TM6 and inward rearrangement of TM7, opening the intracellular G-protein binding pocket for downstream $G_s$ / $G_i$ signaling cascades.
    </div>
    """, unsafe_allow_html=True)
