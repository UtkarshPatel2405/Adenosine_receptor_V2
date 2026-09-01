"""Tab 8: Provenance, Model Integrity & Cryptographic Checksums."""
import pandas as pd
import streamlit as st


def render_tab_provenance(data: dict) -> None:
    st.markdown("""
    <div class="cadd-card">
        <div class="section-num">10</div>
        <div class="section-title" style="color:var(--text-muted)">Provenance & Model Integrity Audit</div>
        <div class="section-subtitle">Cryptographic SHA-256 fingerprints verifying training artifacts and reproducible pipeline outputs</div>
    </div>
    """, unsafe_allow_html=True)

    prov = data.get("provenance", {})
    if not prov:
        st.info("Provenance cryptographic hashes not generated for this payload.")
        return

    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Pipeline Version</div><div class="kpi-value" style="color:var(--cyan);font-size:1rem">{prov.get("version", "2.4.0-precise")}</div></div>', unsafe_allow_html=True)
    with meta_cols[1]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Timestamp</div><div class="kpi-value" style="color:var(--purple);font-size:0.95rem">{prov.get("timestamp", "N/A")}</div></div>', unsafe_allow_html=True)
    with meta_cols[2]:
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">Training Run ID</div><div class="kpi-value" style="color:var(--green);font-size:0.95rem">{prov.get("run_id", "prod_v2")}</div></div>', unsafe_allow_html=True)

    hashes = prov.get("model_hashes", {})
    if hashes:
        st.markdown("<div style='font-size:0.82rem;font-weight:700;color:#f8fafc;margin:1rem 0 0.4rem'>Model Artifact SHA-256 Checksums:</div>", unsafe_allow_html=True)
        hash_rows = [{"Model Artifact": name, "SHA-256 Fingerprint": f'<code style="color:#38bdf8">{h}</code>'} for name, h in hashes.items()]
        st.markdown(pd.DataFrame(hash_rows).to_html(escape=False, index=False, classes="cadd-table", border=0), unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-callout">
        <h4>Theory: Reproducibility & Cryptographic Provenance</h4>
        To guarantee scientific auditability and regulatory compliance, all model binaries, feature scalers, and training matrices are tracked with deterministic SHA-256 digests. This prevents silent model drift and verifies that inference strictly reflects the peer-reviewed training dataset.
    </div>
    """, unsafe_allow_html=True)
