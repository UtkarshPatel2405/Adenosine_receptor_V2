"""Tab: Batch Virtual Screening & Chemical Library Profiler."""
import io
import pandas as pd
import streamlit as st
from src.predictor import predict


def render_tab_batch() -> None:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <h1 class="page-title" style="font-size:1.8rem;margin:0;color:#f8fafc"><span class="material-symbols-outlined">batch_prediction</span>Batch Virtual Screening & Library Profiler</h1>
        <div style="font-size:0.85rem;color:#94a3b8">Screen high-throughput chemical libraries against A1, A2A, A2B, and A3 receptors with multi-subtype selectivity profiles and exportable CSV/SDF datasets</div>
    </div>
    """, unsafe_allow_html=True)

    default_batch_smiles = (
        "CNC(=O)c1cnn(c1)-c1nc(N)c2ncn([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1\n"
        "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O\n"
        "CCN1C(=O)C2=C(N=C(N2C)/C=C/c3ccc(OC)c(OC)c3)N(C1=O)CC\n"
        "CCCn1c(=O)c2[nH]c(-c3ccc(cc3)S(=O)(=O)N3CCN(c4ccc(Cl)cc4)CC3)nc2c(=O)n1CCC\n"
        "Nc1nc(NCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12"
    )

    c_in1, c_in2 = st.columns([1.5, 1.0])
    with c_in1:
        batch_input = st.text_area("Paste SMILES (one per line)", value=default_batch_smiles, height=140, help="Paste list of canonical SMILES strings")
    with c_in2:
        uploaded_file = st.file_uploader("Or Upload Chemical Library (CSV / TXT / TSV)", type=["csv", "txt", "tsv"], help="CSV file must contain a 'smiles' or 'SMILES' column")

    col_btn, col_thresh = st.columns([1.2, 2.8])
    with col_btn:
        run_batch = st.button("🚀 Run Batch Virtual Screen", type="primary", use_container_width=True)
    with col_thresh:
        hit_thresh = st.slider("Activity Hit Threshold (pChEMBL)", min_value=5.0, max_value=9.0, value=6.0, step=0.1, help="Compounds with pChEMBL >= threshold are classified as active hits")

    if run_batch:
        smiles_list = []
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".tsv"):
                    df_up = pd.read_csv(uploaded_file, sep="\t")
                else:
                    df_up = pd.read_csv(uploaded_file)
                smi_col = None
                for c in df_up.columns:
                    if c.strip().lower() in ["smiles", "smi", "canonical_smiles", "structure"]:
                        smi_col = c
                        break
                if smi_col:
                    smiles_list = df_up[smi_col].dropna().astype(str).str.strip().tolist()
                else:
                    smiles_list = df_up.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            except Exception as ex:
                st.error(f"Error parsing uploaded file: {ex}")
                return
        elif batch_input:
            smiles_list = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]

        if not smiles_list:
            st.warning("Please provide at least one valid SMILES string.")
            return

        progress_bar = st.progress(0, text="Screening library against 4 GPCR subtypes...")
        results = []
        n_total = len(smiles_list)
        
        for idx, smi in enumerate(smiles_list):
            progress_bar.progress((idx + 1) / n_total, text=f"Scoring compound {idx+1}/{n_total}...")
            try:
                res = predict(smi, threshold=hit_thresh)
                xgb = res.get("predictions", {}).get("XGBoost", {})
                spectrum = res.get("selectivity_spectrum", {})
                a1 = round(float(xgb.get('A1', 0)), 2)
                a2a = round(float(xgb.get('A2A', 0)), 2)
                a2b = round(float(xgb.get('A2B', 0)), 2)
                a3 = round(float(xgb.get('A3', 0)), 2)
                max_aff = max(a1, a2a, a2b, a3)
                is_hit = max_aff >= hit_thresh
                
                results.append({
                    "Rank": idx + 1,
                    "SMILES": smi,
                    "Primary Target": res.get("best_target", "N/A"),
                    "A1 pChEMBL": a1,
                    "A2A pChEMBL": a2a,
                    "A2B pChEMBL": a2b,
                    "A3 pChEMBL": a3,
                    "Max Affinity": max_aff,
                    "Selectivity Profile": spectrum.get("classification", "N/A"),
                    "Hit Status": "Active Hit" if is_hit else "Inactive / Weak",
                    "In Database": "Yes (Reference)" if res.get("in_database") else "No (De Novo)",
                })
            except Exception as e:
                results.append({
                    "Rank": idx + 1,
                    "SMILES": smi,
                    "Primary Target": "Error",
                    "A1 pChEMBL": 0.0, "A2A pChEMBL": 0.0, "A2B pChEMBL": 0.0, "A3 pChEMBL": 0.0,
                    "Max Affinity": 0.0,
                    "Selectivity Profile": str(e),
                    "Hit Status": "Failed",
                    "In Database": "—",
                })

        progress_bar.empty()
        df_res = pd.DataFrame(results)

        # Screening Summary KPI Strip
        n_hits = sum(1 for r in results if r.get("HitStatus") == "Active Hit" or r.get("Max Affinity", 0) >= hit_thresh)
        avg_aff = sum(r.get("Max Affinity", 0) for r in results) / max(len(results), 1)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Library Size</div><div class="kpi-value" style="color:var(--cyan);font-size:1.25rem">{n_total}</div><div style="font-size:0.75rem;color:var(--text-muted)">Molecules Screened</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Active Hits</div><div class="kpi-value" style="color:var(--green);font-size:1.25rem">{n_hits} / {n_total}</div><div style="font-size:0.75rem;color:var(--text-muted)">pChEMBL ≥ {hit_thresh}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Hit Rate</div><div class="kpi-value" style="color:var(--purple);font-size:1.25rem">{n_hits/max(n_total,1)*100:.1f}%</div><div style="font-size:0.75rem;color:var(--text-muted)">Enrichment Ratio</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Mean Top Affinity</div><div class="kpi-value" style="color:var(--amber);font-size:1.25rem">{avg_aff:.2f}</div><div style="font-size:0.75rem;color:var(--text-muted)">Average pChEMBL</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.88rem;font-weight:700;color:#f8fafc;margin:1.2rem 0 0.4rem'>Scored Chemical Library Registry:</div>", unsafe_allow_html=True)
        st.dataframe(
            df_res,
            column_config={
                "Max Affinity": st.column_config.ProgressColumn("Max Affinity (pChEMBL)", min_value=4.0, max_value=10.0, format="%.2f"),
                "Hit Status": st.column_config.TextColumn("Hit Status"),
            },
            width="stretch",
            hide_index=True,
        )

        # Dual CSV and SDF Export Section
        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.4rem'>Export Scored Screening Results:</div>", unsafe_allow_html=True)
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            csv_buffer = io.StringIO()
            df_res.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Scored Library (CSV - All Metrics)",
                csv_buffer.getvalue(),
                file_name="adenosine_library_screen_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d_c2:
            # Generate concatenated SDF with embedded properties
            sdf_lines = []
            for r in results:
                smi = r.get("SMILES", "")
                if smi:
                    sdf_lines.append(f"{smi}\n> <PRIMARY_TARGET>\n{r.get('Primary Target')}\n\n> <MAX_PCHEMBL>\n{r.get('Max Affinity')}\n\n> <SELECTIVITY_PROFILE>\n{r.get('Selectivity Profile')}\n\n$$$$\n")
            st.download_button(
                "📥 Download Scored Library (SDF / Structure File)",
                "".join(sdf_lines),
                file_name="adenosine_library_screen_results.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True,
            )

