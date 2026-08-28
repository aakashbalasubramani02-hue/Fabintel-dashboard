"""
FABINTEL — Model Registry Page
Displays metadata and performance of deployed models.
"""
import streamlit as st
from components.ui_helpers import render_header, badge, kv_pair, panel, caution


def render():
    render_header()

    st.markdown("# Model Registry")

    # ── Wafer Model ───────────────────────────────────────
    st.markdown("## Wafer Model")

    from inference.wafer_inference import get_model_info
    info = get_model_info()

    w1, w2 = st.columns(2)
    with w1:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">MODEL CONFIGURATION</div>
            {kv_pair("Model", info['name'])}
            {kv_pair("Architecture", info['architecture'])}
            {kv_pair("Task", "Wafer Defect Classification (9 classes)")}
            {kv_pair("Input Resolution", info['input_resolution'])}
            {kv_pair("Loss Function", info['loss'])}
            {kv_pair("Artifact", f'<span class="fi-mono">{info["artifact"]}</span>')}
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    with w2:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">OFFICIAL TEST PERFORMANCE</div>
            {kv_pair("Test Accuracy", info['test_accuracy'])}
            {kv_pair("Test Macro F1", info['test_macro_f1'])}
            {kv_pair("Test Balanced Accuracy", info['test_balanced_accuracy'])}
            <div class="fi-kv-label">Status</div>
            <div class="fi-kv-value">{badge(info['status'], 'ready')}</div>
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    st.markdown(caution(
        "Test metrics were obtained from a single evaluation on the locked WM-811K test set. "
        "The test set is permanently closed and cannot be used for further tuning."
    ), unsafe_allow_html=True)

    st.markdown("---")

    # ── Process Model ─────────────────────────────────────
    st.markdown("## Process Model")

    from inference.secom_inference import get_model_info as get_secom_info
    sinfo = get_secom_info()

    p1, p2 = st.columns(2)
    with p1:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">MODEL CONFIGURATION</div>
            {kv_pair("Model", sinfo['name'])}
            {kv_pair("Task", sinfo['task'])}
            {kv_pair("Input Features", sinfo['input_features'])}
            {kv_pair("Original Features", sinfo['original_features'])}
            {kv_pair("Validation Strategy", sinfo['validation_strategy'])}
            {kv_pair("Artifact", f'<span class="fi-mono">{sinfo["artifact"]}</span>')}
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    with p2:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">CHRONOLOGICAL TEST PERFORMANCE</div>
            {kv_pair("Test Accuracy", sinfo['test_accuracy'])}
            {kv_pair("Test Fail F1", sinfo['test_fail_f1'])}
            {kv_pair("Test ROC-AUC", sinfo['test_roc_auc'])}
            {kv_pair("Test PR-AUC", sinfo['test_pr_auc'])}
            <div class="fi-kv-label">Status</div>
            <div class="fi-kv-value">{badge(sinfo['status'], 'warning')}</div>
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    st.markdown(caution(
        f"KNOWN LIMITATION — {sinfo['limitation']}"
    ), unsafe_allow_html=True)
