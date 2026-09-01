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

    col_moa, col_cascade = st.columns([1, 1])
    with col_moa:
        moa_title = eff.get("mode_of_action", "Unknown")
        moa_color = "var(--green)" if "Agonist" in moa_title else "var(--cyan)" if "Antagonist" in moa_title else "var(--amber)"
        act_prob = eff.get("activation_probability", 0.0)

        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Predicted Mode of Action</div>
            <div class="kpi-value" style="color:{moa_color};font-size:1.1rem">{moa_title}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem">Receptor Activation Probability: <b>{act_prob*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cadd-card" style="margin-top:0.8rem">
            <div style="font-size:0.82rem;font-weight:700;color:#f8fafc;margin-bottom:0.4rem">Target Therapeutic Indication:</div>
            <div style="font-size:0.82rem;color:#cbd5e1">{eff.get("therapeutic_indication", "N/A")}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cascade:
        st.markdown(f"""
        <div class="cadd-card">
            <div style="font-size:0.82rem;font-weight:700;color:var(--cyan);margin-bottom:0.4rem">Downstream Signaling Cascade:</div>
            <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.4">{eff.get("signaling_pathway", "N/A")}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:0.8rem 0 0.4rem'>3D Orthosteric Pocket Interaction Anchors & Stereocenters:</div>", unsafe_allow_html=True)
    i_cols = st.columns(3)
    with i_cols[0]:
        asn_ok = pkt.get("asn_anchor_hbond", False)
        asn_col = "var(--green)" if asn_ok else "var(--red)"
        asn_lbl = "Formed (Dual H-Bond)" if asn_ok else "Missing Anchor"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">{pkt.get("asn_residue", "Asn6.55 Anchor")}</div><div class="kpi-value" style="color:{asn_col};font-size:0.95rem">{asn_lbl}</div></div>', unsafe_allow_html=True)
    with i_cols[1]:
        trp_ok = pkt.get("trp_toggle_switch", False)
        trp_col = "var(--green)" if trp_ok else "var(--amber)"
        trp_lbl = "Active Engagement" if trp_ok else "Inactive Rotamer"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">{pkt.get("trp_residue", "Trp6.48 Toggle")}</div><div class="kpi-value" style="color:{trp_col};font-size:0.95rem">{trp_lbl}</div></div>', unsafe_allow_html=True)
    with i_cols[2]:
        chiral_n = pkt.get("stereocenter_count", 0)
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Chiral Stereocenters</div><div class="kpi-value" style="color:var(--cyan);font-size:0.95rem">{chiral_n} Defined Centers</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="theory-callout"><h4>Stereochemical & Pocket Activation Insight</h4>{pkt.get("chiral_alert", "")}</div>', unsafe_allow_html=True)
