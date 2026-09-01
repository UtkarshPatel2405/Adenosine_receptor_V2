"""Adenosine Receptor Profiler - Industrial CADD & Conformal AI Platform."""
from rdkit import Chem
import streamlit as st

from src.predictor import predict
from src.chem_utils import (
    draw_2d_svg, generate_sdf_block, generate_pdb_block,
    mol_from_smiles, qed_profile, check_pains,
)
from src.api_routes.analysis import receptor_neighbors
from src.provenance import provenance_payload

from src.ui.styles import apply_custom_styles
from src.ui.presets import PRESETS
from src.ui.tab_overview import render_tab_overview
from src.ui.tab_structure import render_tab_structure
from src.ui.tab_selectivity import render_tab_selectivity
from src.ui.tab_efficacy import render_tab_efficacy
from src.ui.tab_safety import render_tab_safety
from src.ui.tab_druglikeness import render_tab_druglikeness
from src.ui.tab_neighbors import render_tab_neighbors
from src.ui.tab_shap import render_tab_shap
from src.ui.tab_structural import render_tab_structural
from src.ui.tab_provenance import render_tab_provenance
from src.ui.tab_batch import render_tab_batch
from src.ui.tab_benchmarks import render_tab_benchmarks
from src.ui.tab_gallery import render_tab_gallery

st.set_page_config(page_title="Adenosine Receptor Profiler", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
apply_custom_styles()

# Sidebar: Platform Information & Methodology Badges
st.sidebar.markdown("### 🧬 Adenosine Profiler")
st.sidebar.markdown("<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:1rem'>State-of-the-Art GPCR Selectivity & Conformal AI Platform</div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(56,189,248,0.2);border-radius:8px;padding:0.75rem;font-size:0.75rem;color:#cbd5e1;line-height:1.5">
    <b>Methodology Highlights:</b><br>
    • <b>4-Subtype Covariance</b>: Multi-target 7-TM GPCR manifold<br>
    • <b>Mode of Action (MoA)</b>: Full/Partial Agonist vs Antagonist<br>
    • <b>Adaptive Conformal</b>: 90% finite-sample uncertainty bounds<br>
    • <b>OECD Principle 3 AD</b>: Tanimoto & physicochemical domain gates<br>
    • <b>Authentic PDBs</b>: Verified GPCRdb.org crystallographic records
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_single, tab_batch, tab_benchmark, tab_gallery = st.tabs([
    "Single Molecule Profiler",
    "Batch Virtual Screening",
    "Model Benchmark Suite",
    "Structural Biology Gallery",
])

with tab_single:
    # Preset Selection Quick-Bar
    p_col1, p_col2 = st.columns([1.2, 2.8])
    with p_col1:
        preset_names = ["Custom Input"] + list(PRESETS.keys())
        selected_preset = st.selectbox("Quick-Load Reference Drug Preset", preset_names, index=1, help="Select a canonical agonist, antagonist, or benchmark molecule")
    with p_col2:
        default_smiles = PRESETS[selected_preset]["smiles"] if selected_preset != "Custom Input" else "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O"
        st.markdown(f"<div style='font-size:0.78rem;color:#94a3b8;margin-top:1.8rem'>Loaded Preset: <b>{selected_preset}</b></div>", unsafe_allow_html=True)

    # Top Hero Command Center Form (Gated with Predict Button / Enter Key)
    with st.form(key="profiler_input_form"):
        st.markdown("""
        <div style="background:linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,41,59,0.7));border:1px solid rgba(56,189,248,0.3);border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;">
            <div style="font-size:1.05rem;font-weight:700;color:#f8fafc">Molecular Structure Input & Activity Threshold</div>
            <div style="font-size:0.78rem;color:#94a3b8">Enter SMILES and click 'Run Selectivity & Profiling Suite' (or press Enter) to execute multi-target prediction</div>
        </div>
        """, unsafe_allow_html=True)

        f_cols = st.columns([3.0, 1.2])
        with f_cols[0]:
            smiles_input = st.text_input("Query Molecule SMILES", value=default_smiles, help="Standard or canonical isomeric SMILES representation")
        with f_cols[1]:
            hit_threshold = st.slider("Activity Hit Threshold (pChEMBL)", 5.0, 9.0, 6.0, 0.1, help="pChEMBL >= 6.0 indicates sub-micromolar active binding")

        submitted = st.form_submit_button("🚀 Run Selectivity & Profiling Suite", type="primary", use_container_width=True)

    # Strictly execute inference ONLY when user submits form
    if submitted:
        curr_smi = smiles_input.strip()
        if not curr_smi:
            st.warning("Please enter a valid SMILES string.")
        else:
            with st.spinner("Executing Multi-Model GPCR Profiling Pipeline..."):
                try:
                    res = predict(curr_smi, threshold=hit_threshold)
                    mol = Chem.MolFromSmiles(res["smiles"])
                    res["svg_2d"] = draw_2d_svg(res["smiles"])
                    res["mol_block_2d"] = Chem.MolToMolBlock(mol) if mol else None
                    res["mol_block_3d"] = generate_sdf_block(res["smiles"])
                    res["pdb_block_3d"] = generate_pdb_block(res["smiles"])
                    res["qed_profile"] = qed_profile(res["smiles"])
                    res["pains_alerts"] = check_pains(res["smiles"])
                    res["receptors"] = {"neighbors": {st_code: receptor_neighbors(res["smiles"], st_code, top_k=5) for st_code in ["A1", "A2A", "A2B", "A3"]}}
                    res["provenance"] = provenance_payload()
                    st.session_state["active_result"] = res
                except Exception as e:
                    st.error(f"Prediction Pipeline Execution Error: {e}")

    if "active_result" in st.session_state:
        res = st.session_state["active_result"]
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
            "Overview & Ki", "2D/3D Conformer", "4-Subtype Selectivity",
            "Mode of Action (MoA)", "Safety & CNS-MPO", "Drug-Likeness (QED)",
            "Chemical Space", "Explainable AI (SHAP)", "Pocket Biology & GPCRdb",
            "Provenance Audit",
        ])
        with t1: render_tab_overview(res)
        with t2: render_tab_structure(res)
        with t3: render_tab_selectivity(res)
        with t4: render_tab_efficacy(res)
        with t5: render_tab_safety(res)
        with t6: render_tab_druglikeness(res)
        with t7: render_tab_neighbors(res)
        with t8: render_tab_shap(res)
        with t9: render_tab_structural(res)
        with t10: render_tab_provenance(res)
    elif not submitted:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);border:1px dashed rgba(56,189,248,0.3);border-radius:10px;padding:2.5rem 1.5rem;text-align:center;color:#94a3b8;margin-top:1rem;">
            <div style="font-size:2.2rem;margin-bottom:0.5rem">🧪</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;margin-bottom:0.3rem">Awaiting Target Molecular Structure</div>
            <div style="font-size:0.85rem;color:#cbd5e1;max-width:520px;margin:0 auto;line-height:1.5;">
                Select a reference drug preset above or enter custom SMILES, then click <b>'Run Selectivity & Profiling Suite'</b> (or press Enter) to launch full 4-subtype affinity, efficacy, safety, and 3D pocket analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_batch:
    render_tab_batch()

with tab_benchmark:
    render_tab_benchmarks()

with tab_gallery:
    render_tab_gallery()
