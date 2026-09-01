"""Tab: Batch Virtual Screening & Library Profiler."""
import io
import pandas as pd
import streamlit as st
from src.predictor import predict


def render_tab_batch() -> None:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <h1 class="page-title" style="font-size:2rem;margin:0;color:#f8fafc"><span class="material-symbols-outlined">batch_prediction</span>Batch Virtual Screening Pipeline</h1>
        <div style="font-size:0.9rem;color:#94a3b8">Screen custom chemical libraries against A1, A2A, A2B, and A3 receptors with conformal confidence intervals</div>
    </div>
    """, unsafe_allow_html=True)

    default_batch_smiles = (
        "CNC(=O)c1cnn(c1)-c1nc(N)c2ncn([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1\n"
        "CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)nc(NCCc4ccc(CCC(=O)O)cc4)nc32)[C@H](O)[C@@H]1O\n"
        "CCN1C(=O)C2=C(N=C(N2C)/C=C/c3ccc(OC)c(OC)c3)N(C1=O)CC\n"
        "CCCn1c(=O)c2[nH]c(-c3ccc(cc3)S(=O)(=O)N3CCN(c4ccc(Cl)cc4)CC3)nc2c(=O)n1CCC\n"
        "Nc1nc(NCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12"
    )

    batch_input = st.text_area("Paste SMILES (one per line)", value=default_batch_smiles, height=150)
    col_btn, col_thresh = st.columns([1, 2])
    with col_btn:
        run_batch = st.button("Run Batch Screen", type="primary", use_container_width=True)
    with col_thresh:
        hit_thresh = st.slider("Activity Hit Threshold (pChEMBL)", min_value=5.0, max_value=9.0, value=6.0, step=0.1)

    if run_batch and batch_input:
        lines = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
        if not lines:
            st.warning("Please provide at least one valid SMILES string.")
            return

        progress_bar = st.progress(0)
        results = []
        for idx, smi in enumerate(lines):
            progress_bar.progress((idx + 1) / len(lines))
            try:
                res = predict(smi, threshold=hit_thresh)
                xgb = res.get("predictions", {}).get("XGBoost", {})
                spectrum = res.get("selectivity_spectrum", {})
                results.append({
                    "SMILES": smi,
                    "Primary Target": res.get("best_target", "N/A"),
                    "A1 pChEMBL": round(float(xgb.get('A1', 0)), 2),
                    "A2A pChEMBL": round(float(xgb.get('A2A', 0)), 2),
                    "A2B pChEMBL": round(float(xgb.get('A2B', 0)), 2),
                    "A3 pChEMBL": round(float(xgb.get('A3', 0)), 2),
                    "Selectivity Profile": spectrum.get("classification", "N/A"),
                    "In Database": "Yes" if res.get("in_database") else "No (De Novo)",
                    "Status": "Valid",
                })
            except Exception as e:
                results.append({
                    "SMILES": smi, "Primary Target": "Error", "A1 pChEMBL": "—", "A2A pChEMBL": "—",
                    "A2B pChEMBL": "—", "A3 pChEMBL": "—", "Selectivity Profile": str(e),
                    "In Database": "—", "Status": "Failed",
                })

        df_res = pd.DataFrame(results)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_res, width="stretch", hide_index=True)

        csv_buffer = io.StringIO()
        df_res.to_csv(csv_buffer, index=False)
        st.download_button("Download Screening Results (CSV)", csv_buffer.getvalue(), file_name="batch_screening_results.csv", mime="text/csv")
