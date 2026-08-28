"""
FABINTEL — Risk Investigation Page
Evidence-based process-risk assessment with SHAP drill-down.
"""
import streamlit as st
import numpy as np
import pandas as pd
from components.ui_helpers import render_header, badge, kv_pair, panel, caution, empty_state, risk_badge
from inference.secom_inference import get_global_shap_ranking, get_model_info
import plotly.graph_objects as go


def render():
    render_header()

    st.markdown("# Risk Investigation")
    st.markdown("Evidence-based process-risk assessment")

    pr = st.session_state.get('current_process_result')
    if pr is None:
        empty_state("🔍", "NO PROCESS ANALYSIS AVAILABLE", "Run a process analysis first from the Process Analytics page.")
        return

    results = pr['results']
    X_imp = pr['X_imp']

    # ── Select Record ─────────────────────────────────────
    record_idx = st.selectbox("Select Record", list(range(len(results))), key="risk_record")
    r = results[record_idx]
    info = get_model_info()

    # ── Prediction Summary ────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        pred_type = 'fail' if r['prediction'] == 'FAIL' else 'pass'
        st.markdown(f"### Predicted Outcome\n{badge(r['prediction'], pred_type)}", unsafe_allow_html=True)
    with sc2:
        st.metric("Failure Probability", f"{r['failure_probability']:.4f}")
    with sc3:
        st.markdown(f"### Model\n`{info['name']}`")

    st.markdown("---")

    # ── Top Process-Risk Factors ──────────────────────────
    st.markdown("## Top Process-Risk Factors")

    with st.spinner("Computing SHAP for selected record..."):
        try:
            from inference.secom_inference import compute_shap, get_top_shap_features
            shap_values, feature_cols, base_value = compute_shap(X_imp[record_idx:record_idx+1])
            top_features = get_top_shap_features(shap_values[0], feature_cols, top_n=20)

            # Bar chart
            fig = go.Figure()
            feat_names = [f['feature'] for f in reversed(top_features)]
            shap_vals = [f['shap_value'] for f in reversed(top_features)]
            colors = ['#c62828' if v > 0 else '#2a6cb6' for v in shap_vals]

            fig.add_trace(go.Bar(
                y=feat_names, x=shap_vals,
                orientation='h',
                marker_color=colors,
                text=[f'{v:+.4f}' for v in shap_vals],
                textposition='outside',
                textfont=dict(size=10, family='JetBrains Mono'),
            ))
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="SHAP contribution",
                font=dict(family='Inter', size=11, color='#3d4f5f'),
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                xaxis=dict(gridcolor='#eef0f2', zeroline=True, zerolinecolor='#1a2332', zerolinewidth=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Ranked table
            st.markdown("### Factor Detail")
            rows = []
            for f in top_features:
                importance = 'HIGH' if f['abs_shap'] > 0.3 else ('MEDIUM' if f['abs_shap'] > 0.1 else 'LOW')
                rows.append({
                    'Feature': f['feature'],
                    'SHAP Contribution': f'{f["shap_value"]:+.4f}',
                    'Relative Importance': importance,
                    'Direction': f['direction'],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"SHAP computation failed: {e}")

    # ── Engineering Review ────────────────────────────────
    st.markdown("## Engineering Review")

    if len(top_features) > 0:
        top = top_features[0]
        st.markdown(f"""
        **{top['feature']}** has a strong model-associated contribution to the current failure prediction.
        The corresponding manufacturing measurement should be reviewed for drift, abnormal variation,
        or out-of-spec behavior.
        """)

    # ── Cross-Module Context ──────────────────────────────
    st.markdown("## System Investigation")

    inv1, inv2 = st.columns(2)

    with inv1:
        st.markdown("### Wafer Intelligence")
        wr = st.session_state.get('current_wafer_result')
        if wr:
            st.markdown(f"**Defect:** {wr['result']['class']}")
            st.markdown(f"**Confidence:** {wr['result']['confidence']:.1%}")
            st.markdown(f"**Grad-CAM:** Available")
        else:
            st.caption("No wafer analysis in current session.")

    with inv2:
        st.markdown("### Process Intelligence")
        st.markdown(f"**Failure risk:** {risk_badge(r['risk'])}", unsafe_allow_html=True)
        st.markdown(f"**Top process-risk factor:** {top_features[0]['feature'] if top_features else '—'}")

    st.markdown("---")
    st.markdown("### Data Relationship")
    st.info("These analyses originate from complementary datasets and are not directly linked at individual wafer level.")

    st.markdown(caution(
        "CAUTION: Statistical/model association does not establish physical causality."
    ), unsafe_allow_html=True)
