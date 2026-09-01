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

# Executive Main Body Hero Header with Glowing Methodology Chips
st.markdown("""
<div style="margin-bottom:1.5rem">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem;border-bottom:1px solid rgba(56,189,248,0.2);padding-bottom:1rem">
        <div>
            <div style="display:flex;align-items:center;gap:0.6rem">
                <span style="font-size:2.2rem">🧬</span>
                <span style="font-family:'Outfit',sans-serif;font-size:2.2rem;font-weight:800;letter-spacing:-0.02em;background:linear-gradient(135deg, #f8fafc, #38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
                    Adenosine Receptor Profiler
                </span>
                <span class="badge-pill badge-cyan" style="font-size:0.75rem;padding:0.2rem 0.6rem">v2.4.0 Production</span>
            </div>
            <div style="font-size:0.92rem;color:#94a3b8;margin-top:0.2rem">
                Industrial CADD & Conformal AI Platform for Multi-Target 7-TM GPCR Selectivity & Drug Discovery
            </div>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center">
            <span class="badge-pill badge-green">✓ 4 GPCR Subtypes</span>
            <span class="badge-pill badge-purple">🛡️ 90% Conformal Validity</span>
            <span class="badge-pill badge-cyan">💎 GPCRdb PDB Co-Crystals</span>
        </div>
    </div>
    <div class="hero-strip" style="margin-top:0.8rem">
        <div class="hero-chip">
            <div class="chip-label">Covariance Regularization</div>
            <div class="chip-value" style="color:var(--cyan);font-size:1.05rem">4-Subtype Manifold</div>
        </div>
        <div class="hero-chip">
            <div class="chip-label">Pharmacological MoA</div>
            <div class="chip-value" style="color:var(--green);font-size:1.05rem">Agonist vs Antagonist</div>
        </div>
        <div class="hero-chip">
            <div class="chip-label">Adaptive Conformal</div>
            <div class="chip-value" style="color:var(--purple);font-size:1.05rem">90% Finite-Sample</div>
        </div>
        <div class="hero-chip">
            <div class="chip-label">OECD Principle 3 AD</div>
            <div class="chip-value" style="color:var(--amber);font-size:1.05rem">Tanimoto + Physicochem</div>
        </div>
        <div class="hero-chip">
            <div class="chip-label">Authentic Structures</div>
            <div class="chip-value" style="color:var(--cyan);font-size:1.05rem">GPCRdb.org Verified</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Quick Navigation & Platform Info
st.sidebar.markdown("### 🧬 Platform Navigation")
st.sidebar.markdown("<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:1rem'>Select preset drug standards or configure virtual screening parameters.</div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="cadd-card" style="padding:0.9rem;font-size:0.75rem;line-height:1.5">
    <b>Quick Reference Standards:</b><br>
    • <b>CGS-21680</b>: A2A Agonist (Ki = 5.7 nM)<br>
    • <b>ZM-241385</b>: A2A Antagonist (Ki = 1.6 nM)<br>
    • <b>PSB-603</b>: A2B Antagonist (Ki = 0.55 nM)<br>
    • <b>CCPA</b>: A1 Selective Agonist (Ki = 0.83 nM)<br>
    • <b>IB-MECA</b>: A3 Agonist (Ki = 1.1 nM)
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_single, tab_batch, tab_benchmark, tab_gallery = st.tabs([
    ":material/science: Single Molecule Profiler",
    ":material/batch_prediction: Batch Virtual Screening",
    ":material/analytics: Model Benchmark Suite",
    ":material/view_in_ar: Structural Biology Gallery",
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
            ":material/dashboard: Overview & Ki",
            ":material/view_in_ar: 2D/3D Conformer",
            ":material/radar: Selectivity Radar",
            ":material/vital_signs: Efficacy & MoA",
            ":material/health_and_safety: Safety & CNS-MPO",
            ":material/medication: Drug-Likeness (QED)",
            ":material/hub: Chemical Space",
            ":material/psychology: Explainable AI (SHAP)",
            ":material/biotech: Pocket Biology & GPCRdb",
            ":material/verified_user: Provenance Audit",
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
