"""Tab 8: TreeSHAP Feature Attributions & Explainable AI."""
import pandas as pd
import plotly.express as px
import streamlit as st
from src.api_routes.analysis import shap_analysis


def render_tab_shap(data: dict) -> None:
    smiles = data.get("smiles", "")
    best_target = data.get("best_target", "A2A")

    st.markdown("""
    <div class="cadd-card">
        <div class="section-title" style="color:var(--purple)">TreeSHAP Feature Attributions & Explainability</div>
        <div class="section-subtitle">Local Shapley Additive Explanations quantifying feature contributions to subtype affinity</div>
    </div>
    """, unsafe_allow_html=True)

    sel_sub = st.segmented_control("Receptor Subtype", ["A1", "A2A", "A2B", "A3"], default=best_target if best_target in ["A1", "A2A", "A2B", "A3"] else "A2A", key="shap_sub", label_visibility="collapsed")
    
    with st.spinner(f"Computing TreeSHAP feature attributions for Human {sel_sub}..."):
        shap_res = shap_analysis(smiles, sel_sub, top_k=10)

    if not shap_res or not shap_res.get("features"):
        st.info(f"TreeSHAP feature attributions not available for Human {sel_sub}.")
        return

    features = [f["feature"] for f in shap_res["features"]]
    values = [f["value"] for f in shap_res["features"]]
    df_shap = pd.DataFrame({"Feature": features, "SHAP Contribution": values}).sort_values(by="SHAP Contribution", key=abs, ascending=True)
    df_shap["Impact"] = df_shap["SHAP Contribution"].apply(lambda v: "Positive (Increases Affinity)" if v > 0 else "Negative (Decreases Affinity)")

    fig = px.bar(
        df_shap,
        x="SHAP Contribution",
        y="Feature",
        color="Impact",
        color_discrete_map={"Positive (Increases Affinity)": "#4ade80", "Negative (Decreases Affinity)": "#f87171"},
        orientation="h",
        title=f"Top 10 Feature Drivers for Human {sel_sub} Affinity (Base Value = {shap_res.get('base_value', 0):.2f})",
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="Inter"),
        xaxis=dict(gridcolor='rgba(216, 224, 230, 0.15)', title="SHAP Value (Δ pChEMBL Impact)"),
        yaxis=dict(title=""),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("""
    <div class="theory-callout">
        <h4>Theory: TreeSHAP Molecular Feature Attributions</h4>
        TreeSHAP computes exact Shapley values from cooperative game theory across tree ensemble paths. Positive SHAP contributions indicate functional groups or physicochemical properties (e.g. aromatic rings, hydrogen bond acceptors) that elevate predicted binding affinity for this specific subtype.
    </div>
    """, unsafe_allow_html=True)
