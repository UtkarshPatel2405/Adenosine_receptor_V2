"""Tab: Structural Biology 3D Pocket Gallery & GPCRdb Reference."""
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from src.ui.presets import GPCRDB_CATALOG_RECORDS
from src.ui.renderers_3d import render_3dmol_complex


def render_tab_gallery() -> None:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <h1 class="page-title" style="font-size:2rem;margin:0;color:#f8fafc"><span class="material-symbols-outlined">view_in_ar</span>Structural Biology 3D Pocket Gallery & GPCRdb Reference</h1>
        <div style="font-size:0.9rem;color:#94a3b8">Verified crystal and Cryo-EM structural complexes across human adenosine receptors (A1, A2A, A2B, A3) from GPCRdb & RCSB</div>
    </div>
    """, unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("""<h3 class="page-title" style="color:var(--green);font-size:1.1rem"><span class="material-symbols-outlined">radio_button_checked</span>Active Signaling Conformations (Agonist-Bound)</h3>""", unsafe_allow_html=True)
        act_sel = st.selectbox(
            "Active Structure",
            [
                "A2A: 6GDG (4.1 Å Cryo-EM, Adenosine–miniGs)",
                "A1: 6D9H (3.6 Å Cryo-EM, Adenosine–Gi2)",
                "A2B: 8HDO (2.87 Å Cryo-EM, BAY 60-6583–Gs)",
                "A3: 8X16 (3.29 Å Cryo-EM, CF101 / IB-MECA–Gi)",
            ],
            key="g_act",
        )
        pdb_act = act_sel.split(":")[1].split("(")[0].strip()
        components.html(render_3dmol_complex(pdb_act), height=430)

    with g_col2:
        st.markdown("""<h3 class="page-title" style="color:var(--red);font-size:1.1rem"><span class="material-symbols-outlined">pause_circle</span>Inactive Ground State Conformations (Antagonist-Bound)</h3>""", unsafe_allow_html=True)
        inact_sel = st.selectbox(
            "Inactive Structure",
            [
                "A2A: 4EIY (1.8 Å X-ray, ZM241385)",
                "A1: 5N2S (3.3 Å X-ray, PSB36)",
                "A3: 9EHS (3.2 Å Cryo-EM, LUF7602)",
                "A2B: None (No Experimental Inactive Structure Solved)",
            ],
            key="g_inact",
        )
        pdb_inact_token = inact_sel.split(":")[1].split("(")[0].strip()
        pdb_inact = None if pdb_inact_token.lower() == "none" else pdb_inact_token
        components.html(render_3dmol_complex(pdb_inact), height=430)

    st.markdown("""
    <div class="cadd-card" style="margin-top:1.5rem">
        <div class="section-num">04b</div>
        <div class="section-title" style="color:var(--cyan)"><span class="material-symbols-outlined">table_chart</span> Curated Human Adenosine GPCR Structural Landscape (GPCRdb / RCSB)</div>
        <div class="section-subtitle">Authoritative catalog of experimental active and inactive structures with resolution and ligand pharmacology</div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(GPCRDB_CATALOG_RECORDS), width="stretch", hide_index=True)
