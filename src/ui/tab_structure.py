"""Tab 2: Molecular 2D Topological Depiction & 3D Conformer Viewer."""
import streamlit as st
import streamlit.components.v1 as components
from src.ui.renderers_3d import render_3dmol_conformer


def render_tab_structure(data: dict) -> None:
    st.markdown("""
    <div class="cadd-card">
        <div class="section-title" style="color:var(--cyan)">Molecular Visualization & Structural Coordinates</div>
        <div class="section-subtitle">2D topological depiction and 3D MMFF94 energy-minimized conformer</div>
    </div>
    """, unsafe_allow_html=True)

    col_v2d, col_v3d = st.columns([1, 1])
    with col_v2d:
        st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#c8d0d6;margin-bottom:0.4rem'>2D Structure (RDKit)</div>", unsafe_allow_html=True)
        if data.get("svg_2d"):
            st.markdown(f"<div style='background:#333f45;border:1px solid rgba(216,224,230,0.15);border-radius:8px;padding:1rem;display:flex;justify-content:center'>{data['svg_2d']}</div>", unsafe_allow_html=True)
        else:
            st.info("2D SVG unavailable.")

    with col_v3d:
        st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#c8d0d6;margin-bottom:0.4rem'>3D MMFF94 Conformer (RDKit ETKDGv3 · Isolated Unbound Ligand)</div>", unsafe_allow_html=True)
        if data.get("mol_block_3d"):
            components.html(render_3dmol_conformer(data["mol_block_3d"]), height=390)
        else:
            st.info("3D conformer unavailable.")

    st.markdown("<div style='margin-top:0.6rem;font-size:0.82rem;font-weight:600;color:#f8fafc'>Download Molecular Coordinates:</div>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
    with dl_col1:
        if data.get("svg_2d"):
            st.download_button("Download 2D SVG", data["svg_2d"], file_name="molecule_2d.svg", mime="image/svg+xml", width="stretch")
    with dl_col2:
        if data.get("mol_block_2d"):
            st.download_button("Download 2D SDF", data["mol_block_2d"], file_name="molecule_2d.sdf", mime="chemical/x-mdl-sdfile", width="stretch")
    with dl_col3:
        if data.get("mol_block_3d"):
            st.download_button("Download 3D SDF", data["mol_block_3d"], file_name="conformer_3d.sdf", mime="chemical/x-mdl-sdfile", width="stretch")
    with dl_col4:
        if data.get("pdb_block_3d"):
            st.download_button("Download 3D PDB", data["pdb_block_3d"], file_name="conformer_3d.pdb", mime="chemical/x-pdb", width="stretch")

    st.markdown("""
    <div class="theory-callout">
        <h4>Theory: Molecular Representation & Force-Field Conformations</h4>
        SMILES encodes chemical topology without 3D coordinate bias. 3D conformers are generated using Experimental-Torsion Distance Geometry with Knowledge (ETKDGv3) and energy-minimized using the MMFF94 force field to capture the spatial orientation and partial charge distribution prior to receptor interaction.
    </div>
    """, unsafe_allow_html=True)
