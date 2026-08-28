"""
FABINTEL — Process Analytics Page
Upload SECOM data → preprocess → predict → SHAP.
"""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from components.ui_helpers import render_header, badge, kv_pair, panel, caution, empty_state, risk_badge
from inference.secom_inference import preprocess_csv, predict_process, compute_shap, get_top_shap_features, load_demo_records, get_model_info
import plotly.graph_objects as go


def render():
    render_header()

    st.markdown("# Process Analytics")
    st.markdown("Manufacturing failure-risk assessment")

    # ── Input ─────────────────────────────────────────────
    col_up, col_demo = st.columns([3, 2])

    with col_up:
        uploaded = st.file_uploader("Upload Process Data (CSV, space/comma-separated)", type=['csv', 'data', 'txt'])
    with col_demo:
        use_demo = st.button("Load Demo Record")

    # ── Determine input ───────────────────────────────────
    input_df = None
    input_desc = "—"

    if uploaded is not None:
        try:
            # Try space-separated first, then comma
            try:
                input_df = pd.read_csv(uploaded, sep=' ', header=None)
            except Exception:
                uploaded.seek(0)
                input_df = pd.read_csv(uploaded, header=None)
            input_desc = f"Uploaded: {uploaded.name}"
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
    elif use_demo:
        input_df = load_demo_records(n=3)
        input_desc = "Demo: SECOM sample records"

    if input_df is None:
        empty_state("⚙", "NO PROCESS DATA", "Upload a process measurement record to begin risk analysis.")
        return

    # ── Preprocess ────────────────────────────────────────
    st.markdown("## Preprocessing")
    try:
        X_imp, meta = preprocess_csv(input_df)
    except ValueError as e:
        st.error(str(e))
        return

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Records", meta['n_records'])
    mc2.metric("Features Used", meta['n_features_used'])
    mc3.metric("Features Total", meta['n_features_total'])
    mc4.metric("Missing Imputed", meta['missing_values_imputed'])

    st.markdown(f'**Model Status:** {badge("READY", "ready")}', unsafe_allow_html=True)

    # ── Predict ───────────────────────────────────────────
    st.markdown("## Process Assessment")

    results = predict_process(X_imp)

    for r in results:
        with st.container():
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                pred_type = 'fail' if r['prediction'] == 'FAIL' else 'pass'
                st.markdown(f"**Record {r['index']}:** {badge(r['prediction'], pred_type)}", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"**Failure probability:** `{r['failure_probability']:.4f}`")
            with rc3:
                st.markdown(f"**Risk:** {risk_badge(r['risk'])}", unsafe_allow_html=True)

        # Add to session history
        st.session_state.session_history.append({
            'type': 'process',
            'time': datetime.now().strftime("%H:%M:%S"),
            'input_desc': input_desc,
            'prediction': r['prediction'],
            'confidence': r['failure_probability'],
            'risk': r['risk'],
            'status': r['risk'],
        })

    # Store for other pages
    st.session_state.current_process_result = {
        'results': results,
        'X_imp': X_imp,
        'meta': meta,
        'input_desc': input_desc,
        'time': datetime.now().strftime("%H:%M:%S"),
    }

    # ── SHAP ──────────────────────────────────────────────
    st.markdown("## Process-Risk Factors")

    shap_tab_global, shap_tab_record = st.tabs(["Global Analysis", "Current Record"])

    with shap_tab_global:
        from inference.secom_inference import get_global_shap_ranking
        global_df = get_global_shap_ranking()
        if global_df is not None:
            fig = go.Figure()
            top = global_df.head(15)
            fig.add_trace(go.Bar(
                y=[f"F{int(r)}" for r in top['Feature']],
                x=top['Mean_Abs_SHAP'],
                orientation='h',
                marker_color='#2a6cb6',
                text=[f'{v:.3f}' for v in top['Mean_Abs_SHAP']],
                textposition='outside',
                textfont=dict(size=10, family='JetBrains Mono'),
            ))
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange='reversed'),
                font=dict(family='Inter', size=11, color='#3d4f5f'),
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                xaxis=dict(gridcolor='#eef0f2'),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Global SHAP ranking not available.")

    with shap_tab_record:
        record_idx = st.selectbox("Select Record", list(range(len(results))))
        with st.spinner("Computing SHAP..."):
            try:
                shap_values, feature_cols, base_value = compute_shap(X_imp[record_idx:record_idx+1])
                top_features = get_top_shap_features(shap_values[0], feature_cols, top_n=15)

                rows = []
                for f in top_features:
                    importance = 'HIGH' if f['abs_shap'] > 0.3 else ('MEDIUM' if f['abs_shap'] > 0.1 else 'LOW')
                    rows.append({
                        'Feature': f['feature'],
                        'SHAP Value': f'{f["shap_value"]:.4f}',
                        'Importance': importance,
                        'Direction': f['direction'],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")

    st.markdown(caution(
        "Statistical/model association does not establish physical causality. "
        "Features listed are candidate process-risk factors requiring engineering investigation."
    ), unsafe_allow_html=True)
