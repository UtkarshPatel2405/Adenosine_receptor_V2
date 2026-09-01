"""Tab 3: 4-Subtype Affinity Spectrum, Radar Polygon & Selectivity Matrix."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_tab_selectivity(data: dict) -> None:
    preds = data.get("predictions", {})
    iv = data.get("intervals", {})
    xgb = preds.get("XGBoost", {})
    db_val = data.get("db_value") or {}
    in_db = data.get("in_database", False)
    subtypes_list = ["A1", "A2A", "A2B", "A3"]

    st.markdown("""
    <div class="cadd-card">
        <div class="section-num">03</div>
        <div class="section-title" style="color:var(--green)">4-Subtype Affinity Spectrum & Selectivity Matrix</div>
        <div class="section-subtitle">Multi-receptor binding polygon, subtype rank hierarchy, and comprehensive pairwise selectivity differentials</div>
    </div>
    """, unsafe_allow_html=True)

    col_radar, col_sel_table = st.columns([1, 1])
    with col_radar:
        radar_subtypes = ["A1", "A2A", "A2B", "A3", "A1"]
        radar_vals = [float(xgb.get(s, 0) or 0) for s in ["A1", "A2A", "A2B", "A3"]] + [float(xgb.get("A1", 0) or 0)]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=radar_vals,
            theta=radar_subtypes,
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38bdf8', width=2),
            name='Predicted pChEMBL'
        ))
        if in_db and any(pd.notna(db_val.get(s)) for s in subtypes_list):
            db_radar = [float(db_val.get(s) or 0) for s in ["A1", "A2A", "A2B", "A3"]] + [float(db_val.get("A1") or 0)]
            fig_radar.add_trace(go.Scatterpolar(
                r=db_radar,
                theta=radar_subtypes,
                fill='none',
                line=dict(color='#86efac', width=2, dash='dash'),
                name='Experimental ChEMBL'
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], color='#9aa7af', gridcolor='rgba(216, 224, 230, 0.15)'),
                angularaxis=dict(color='#eef2f4', gridcolor='rgba(216, 224, 230, 0.15)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=30, r=30, t=20, b=20),
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color="#c8d0d6"))
        )
        st.plotly_chart(fig_radar, width="stretch")

    with col_sel_table:
        st.markdown("<div style='font-size:0.82rem;font-weight:600;color:#f8fafc;margin-bottom:0.4rem'>Target Affinity Hierarchy (Rank-Ordered):</div>", unsafe_allow_html=True)
        sorted_ranks = sorted([s for s in subtypes_list if xgb.get(s) is not None], key=lambda s: float(xgb.get(s, 0) or 0), reverse=True)
        max_v = float(xgb.get(sorted_ranks[0], 0) or 0) if sorted_ranks else 0.0
        
        rank_rows = []
        for r_idx, s in enumerate(sorted_ranks, 1):
            val = float(xgb.get(s, 0) or 0)
            ci = iv.get("XGBoost", {}).get(s, {})
            ci_low = ci.get("lower", val - 0.5)
            ci_high = ci.get("upper", val + 0.5)
            delta_from_top = max_v - val
            fold_from_top = 10 ** delta_from_top
            fold_str = "Primary Target" if r_idx == 1 else f"{fold_from_top:.1f}x lower"
            rank_rows.append({
                "Rank": f"#{r_idx}",
                "Receptor Subtype": f"Human {s}",
                "Predicted pChEMBL": f"{val:.2f} [{ci_low:.2f} – {ci_high:.2f}]",
                "Relative Separation": fold_str,
            })
        st.dataframe(pd.DataFrame(rank_rows), width="stretch", hide_index=True)

    st.markdown("<div style='font-size:0.82rem;font-weight:600;color:#f8fafc;margin:0.8rem 0 0.4rem'>Complete Pairwise Subtype Differentials Matrix (ΔpChEMBL):</div>", unsafe_allow_html=True)
    all_pairs = [("A1", "A2A"), ("A1", "A2B"), ("A1", "A3"), ("A2A", "A2B"), ("A2A", "A3"), ("A2B", "A3")]
    sel_rows = []
    for subA, subB in all_pairs:
        valA = float(xgb.get(subA, 0) or 0)
        valB = float(xgb.get(subB, 0) or 0)
        delta = valA - valB
        fold = 10 ** abs(delta)
        pref = f"{subA} Selective ({fold:.1f}x)" if delta >= 0.5 else f"{subB} Selective ({fold:.1f}x)" if delta <= -0.5 else f"Equipotent ({fold:.1f}x)"
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        sel_rows.append({"Comparison Pair": f"{subA} vs {subB}", "Δ pChEMBL": delta_str, "Selectivity Ratio": f"{fold:.1f}x", "Classification": pref})
    st.dataframe(pd.DataFrame(sel_rows), width="stretch", hide_index=True)
