"""Tab 9: Authentic GPCRdb Pocket Biology & Active/Inactive Structural Explorer."""
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from src.ui.presets import RECEPTOR_STRUCT_DB, GPCRDB_CATALOG_RECORDS
from src.ui.renderers_3d import render_3dmol_complex
from src.pdb_utils import find_gpcrdb_structure_matches


def render_tab_structural(data: dict) -> None:
    smiles = data.get("smiles", "")
    st.markdown("""
    <div class="cadd-card">
        <div class="section-title" style="color:var(--cyan)">Pocket Biology & Structural Conformations</div>
        <div class="section-subtitle">Authentic deposited Cryo-EM & X-ray crystallographic complexes from GPCRdb and RCSB PDB</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. GPCRdb Structural Analog Matches for Query Molecule
    with st.expander("Curated GPCRdb Structural Analog Matches (Tanimoto >= 0.20)", expanded=True):
        gpcr_matches = find_gpcrdb_structure_matches(smiles, min_tanimoto=0.20)
        if gpcr_matches:
            m_rows = []
            for m in gpcr_matches:
                pdb_id = m.get("pdb_id", "")
                gpcr_url = m.get("gpcrdb_url", f"https://gpcrdb.org/structure/{pdb_id}")
                m_rows.append({
                    "PDB Entry": pdb_id,
                    "Receptor Subtype": f"Human {m.get('subtype')}",
                    "Conformation State": m.get("state", "N/A"),
                    "Experimental Resolution": f"{m.get('method', '')} ({m.get('resolution', '')})",
                    "Co-Crystallized Ligand": m.get("ligand_name", "N/A"),
                    "Tanimoto Similarity": round(float(m.get("tanimoto", 0)), 3),
                    "GPCRdb Link": gpcr_url,
                })
            df_matches = pd.DataFrame(m_rows)
            st.dataframe(
                df_matches,
                column_config={
                    "GPCRdb Link": st.column_config.LinkColumn("GPCRdb Entry", display_text="View on GPCRdb.org"),
                },
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No close co-crystallized structural analogs found in GPCRdb above Tanimoto 0.20.")

    # 2. Both Active and Inactive Conformational Catalogs
    with st.expander("Complete Human Adenosine Receptor Crystal / Cryo-EM Catalog (Active & Inactive)", expanded=False):
        cat_rows = []
        for r in GPCRDB_CATALOG_RECORDS:
            pdb_id = r.get("PDB ID", "—")
            link = f"https://gpcrdb.org/structure/{pdb_id}" if pdb_id != "—" else "https://alphafold.ebi.ac.uk/entry/P29275"
            cat_rows.append({
                "Receptor Subtype": r.get("Subtype", ""),
                "Conformation State": r.get("State", ""),
                "PDB ID": pdb_id,
                "Method & Resolution": f"{r.get('Method', '')} ({r.get('Resolution', '')})",
                "Co-Crystal Ligand / Complex": r.get("Ligand / Complex", ""),
                "GPCRdb Structure Link": link,
            })
        st.dataframe(
            pd.DataFrame(cat_rows),
            column_config={
                "GPCRdb Structure Link": st.column_config.LinkColumn("Structure Link", display_text="Open Structure"),
            },
            width="stretch",
            hide_index=True,
        )

    # 3. Interactive 3D PDB Complex Explorer
    st.markdown("<div style='font-size:0.9rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.4rem'>Interactive 3D Receptor-Ligand Complex Viewer:</div>", unsafe_allow_html=True)
    st_cols = st.columns([1, 1])
    with st_cols[0]: sel_subtype = st.selectbox("Select Receptor Subtype", ["A1", "A2A", "A2B", "A3"], index=1, key="struct_sub")
    with st_cols[1]: sel_state = st.selectbox("Select Conformational State", ["Active (Agonist-bound)", "Inactive (Antagonist-bound)"], index=0, key="struct_state")

    state_key = "active" if "Active" in sel_state else "inactive"
    current_pdb_meta = RECEPTOR_STRUCT_DB[sel_subtype][state_key]
    current_pdb_id = current_pdb_meta["pdb_id"]

    col_viz, col_theory = st.columns([1.1, 0.9])
    with col_viz:
        st.markdown(f'<div style="font-size:0.85rem;font-weight:700;color:#38bdf8;margin-bottom:0.4rem">Human {sel_subtype} ({current_pdb_id} - {current_pdb_meta["resolution"]})</div>', unsafe_allow_html=True)
        components.html(render_3dmol_complex(current_pdb_id), height=420)
    with col_theory:
        st.markdown(f"""
        <div class="theory-callout" style="margin-top:0">
            <h4>Experimental Pocket Mechanism</h4>
            <b>{current_pdb_meta['title']}</b><br><br>
            {current_pdb_meta['mechanism']}<br><br>
            <div style="background:rgba(56,189,248,0.08);border-left:2px solid var(--cyan);padding:0.5rem 0.7rem;border-radius:0 4px 4px 0">
                <b style="color:var(--cyan)">CADD Design Insight:</b> {current_pdb_meta['cadd_note']}
            </div>
        </div>
        """, unsafe_allow_html=True)
