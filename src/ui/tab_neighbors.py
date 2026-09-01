"""Tab 7: Training Neighbors & Chemical Space Proximity Analysis."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.chem_utils import draw_2d_svg


def render_tab_neighbors(data: dict) -> None:
    st.markdown("""
    <div class="cadd-card">
        <div class="section-title" style="color:var(--green)">Subtype-Specific Training Neighbors & Chemical Space</div>
        <div class="section-subtitle">Ranked nearest active and reference ligands in the training dataset by Morgan Tanimoto similarity (radius=2, 2048-bit). pChEMBL ≥ 6.0 = Active hit.</div>
    </div>
    """, unsafe_allow_html=True)

    rec_neighbors = data.get("receptors", {}).get("neighbors", {}) or {}
    sel_sub = st.segmented_control("Receptor Subtype", ["A1", "A2A", "A2B", "A3"], default="A2A", label_visibility="collapsed")
    nbrs = rec_neighbors.get(sel_sub, [])

    if not nbrs:
        st.info(f"No per-receptor training neighbors found for Human {sel_sub}.")
        return

    # 1. Proximity Summary KPI Strip
    max_tan = max((float(n.get("tanimoto", 0) or 0) for n in nbrs), default=0.0)
    n_actives = sum(1 for n in nbrs if (n.get("pchembl") or 0) >= 6.0)
    avg_pcm = sum((n.get("pchembl") or 0) for n in nbrs) / max(len(nbrs), 1)
    top_struct = (nbrs[0].get("real_structures") or [{}])[0] if nbrs else {}
    top_pdb = top_struct.get("id", "N/A")

    k_cols = st.columns(4)
    with k_cols[0]:
        tan_col = "var(--green)" if max_tan >= 0.7 else "var(--amber)" if max_tan >= 0.4 else "var(--cyan)"
        tan_lbl = "High Similarity" if max_tan >= 0.7 else "Moderate Analog" if max_tan >= 0.4 else "Scaffold Hop"
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Nearest Tanimoto</div><div class="kpi-value" style="color:{tan_col};font-size:1.15rem">{max_tan*100:.1f}%</div><div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem">{tan_lbl}</div></div>', unsafe_allow_html=True)
    with k_cols[1]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Active Analogs (Top {len(nbrs)})</div><div class="kpi-value" style="color:var(--green);font-size:1.15rem">{n_actives} / {len(nbrs)}</div><div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem">pChEMBL ≥ 6.0 (Ki ≤ 1 µM)</div></div>', unsafe_allow_html=True)
    with k_cols[2]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Mean Analog Affinity</div><div class="kpi-value" style="color:var(--purple);font-size:1.15rem">{avg_pcm:.2f}</div><div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem">Average pChEMBL</div></div>', unsafe_allow_html=True)
    with k_cols[3]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Structural Template</div><div class="kpi-value" style="color:var(--cyan);font-size:1.15rem">{top_pdb}</div><div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem">Closest PDB Co-Crystal</div></div>', unsafe_allow_html=True)

    # 2. Visual Top-3 Nearest Structural Analogs Cards
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.4rem'>Closest Structural Analogs in Training Library:</div>", unsafe_allow_html=True)
    top_3 = nbrs[:3]
    card_cols = st.columns(len(top_3))
    for idx, (col, n) in enumerate(zip(card_cols, top_3), 1):
        with col:
            smi = n.get("smiles", "")
            tan = float(n.get("tanimoto", 0) or 0)
            pcm = float(n.get("pchembl", 0) or 0)
            act = n.get("activity", "Unknown")
            badge_color = "#4ade80" if act == "Active" else "#fbbf24" if act == "Weak" else "#f87171"
            svg_code = draw_2d_svg(smi, width=240, height=160) if smi else None
            
            with st.container(border=True):
                st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem'><span style='font-weight:700;color:#38bdf8'>Rank #{idx}</span><span style='background:rgba(56,189,248,0.15);color:#7dd3fc;padding:0.15rem 0.5rem;border-radius:9999px;font-size:0.75rem;font-weight:600'>{tan*100:.1f}% Match</span></div>", unsafe_allow_html=True)
                if svg_code:
                    st.markdown(f"<div style='background:#1e293b;border-radius:6px;padding:0.4rem;display:flex;justify-content:center;margin-bottom:0.4rem'>{svg_code}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8rem;color:#cbd5e1;display:flex;justify-content:space-between'><span>Assay pChEMBL: <b>{pcm:.2f}</b></span><span style='color:{badge_color};font-weight:600'>{act}</span></div>", unsafe_allow_html=True)

    # 3. Chemical Similarity & Affinity Comparative Ranking Chart
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.4rem'>Training Chemical Space Similarity Spectrum:</div>", unsafe_allow_html=True)
    plot_rows = []
    for idx, n in enumerate(nbrs, 1):
        tan = float(n.get("tanimoto", 0) or 0)
        pcm = float(n.get("pchembl", 0) or 0)
        act = n.get("activity", "Unknown")
        smi = n.get("smiles", "")
        plot_rows.append({
            "Neighbor": f"#{idx} ({tan*100:.1f}%)",
            "Tanimoto Similarity": round(tan, 3),
            "Affinity (pChEMBL)": round(pcm, 2),
            "Activity Status": act,
            "SMILES": smi,
        })
    
    df_plot = pd.DataFrame(plot_rows)
    fig = px.bar(
        df_plot,
        x="Tanimoto Similarity",
        y="Neighbor",
        orientation="h",
        color="Activity Status",
        color_discrete_map={"Active": "#4ade80", "Weak": "#fbbf24", "Inactive": "#f87171"},
        hover_data=["Affinity (pChEMBL)", "SMILES"],
        title=f"Top {len(nbrs)} Nearest Training Analogs Ranked by Structural Proximity (Human {sel_sub})",
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="Inter"),
        xaxis=dict(gridcolor='rgba(216, 224, 230, 0.15)', range=[0, 1.0], tickformat=".0%"),
        yaxis=dict(autorange="reversed", title=""),
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, width="stretch")

    # 4. Detailed Tabular Breakdown with Clean Native Links
    st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:0.8rem 0 0.4rem'>Full Analog Assay & Co-Crystal Registry:</div>", unsafe_allow_html=True)
    rows = []
    for i, n in enumerate(nbrs, 1):
        pcm = n.get("pchembl")
        act = n.get("activity", "—")
        tan = n.get("tanimoto")
        first_struct = (n.get("real_structures") or [{}])[0] if (n.get("real_structures") or []) else {}
        struct_id = first_struct.get("id", "")
        gpcr_link = first_struct.get("gpcrdb_url") or (f"https://gpcrdb.org/structure/{struct_id}" if struct_id else None)
        rows.append({
            "Rank": f"#{i}",
            "Neighbor SMILES": n.get("smiles", ""),
            "Tanimoto Similarity": f"{float(tan)*100:.1f}%" if tan is not None else "0.0%",
            "Experimental pChEMBL": f"{float(pcm):.2f}" if pcm is not None else "—",
            "Activity Status": act,
            "GPCRdb Structural Template": gpcr_link or "https://gpcrdb.org",
        })

    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "GPCRdb Structural Template": st.column_config.LinkColumn("GPCRdb Entry", display_text="View Co-Crystal PDB"),
        },
        width="stretch",
        hide_index=True,
    )

