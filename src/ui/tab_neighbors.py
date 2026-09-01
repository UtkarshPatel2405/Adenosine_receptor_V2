"""Tab 7: Training Neighbors & Interactive Tanimoto Scatter Bubble Chart."""
import pandas as pd
import plotly.express as px
import streamlit as st


def render_tab_neighbors(data: dict) -> None:
    st.markdown("""
    <div class="cadd-card">
        <div class="section-title" style="color:var(--green)">Subtype-Specific Training Neighbors & Chemical Space</div>
        <div class="section-subtitle">Tanimoto similarity (Morgan FP, radius=2). pChEMBL >= 6.0 = Active.</div>
    </div>
    """, unsafe_allow_html=True)

    rec_neighbors = data.get("receptors", {}).get("neighbors", {}) or {}
    sel_sub = st.segmented_control("Receptor Subtype", ["A1", "A2A", "A2B", "A3"], default="A2A", label_visibility="collapsed")
    nbrs = rec_neighbors.get(sel_sub, [])

    if not nbrs:
        st.info(f"No per-receptor training neighbors found for Human {sel_sub}.")
        return

    # 1. Interactive 2D Bubble Scatter Chart (Tanimoto vs pChEMBL)
    plot_data = []
    for n in nbrs:
        pcm = n.get("pchembl") or 5.0
        tan = n.get("tanimoto") or 0.0
        act = n.get("activity", "Unknown")
        plot_data.append({"Tanimoto Similarity": round(tan, 3), "Affinity (pChEMBL)": round(pcm, 2), "Activity Class": act, "SMILES": n.get("smiles", "")[:35] + "..."})

    df_plot = pd.DataFrame(plot_data)
    fig = px.scatter(
        df_plot,
        x="Tanimoto Similarity",
        y="Affinity (pChEMBL)",
        color="Activity Class",
        size="Affinity (pChEMBL)",
        hover_data=["SMILES"],
        color_discrete_map={"Active": "#4ade80", "Weak": "#facc15", "Inactive": "#f87171"},
        title=f"Nearest Training Neighbors in Human {sel_sub} Chemical Space",
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="Inter"),
        xaxis=dict(gridcolor='rgba(216, 224, 230, 0.15)', range=[0, 1.05]),
        yaxis=dict(gridcolor='rgba(216, 224, 230, 0.15)'),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    # 2. Detailed Tabular Breakdown with Clean Native Links
    rows = []
    for i, n in enumerate(nbrs, 1):
        pcm = n.get("pchembl")
        act = n.get("activity", "—")
        tan = n.get("tanimoto")
        first_struct = (n.get("real_structures") or [{}])[0] if (n.get("real_structures") or []) else {}
        struct_id = first_struct.get("id", "")
        gpcr_link = first_struct.get("gpcrdb_url") or (f"https://gpcrdb.org/structure/{struct_id}" if struct_id else None)
        rows.append({
            "Rank": i,
            "Neighbor SMILES": n.get("smiles", ""),
            "Tanimoto Similarity": round(float(tan), 3) if tan is not None else 0.0,
            "Bioactivity (pChEMBL)": round(float(pcm), 2) if pcm is not None else 0.0,
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
