"""
FABINTEL — System Status Page
Component health and environment information.
"""
import os
import streamlit as st
from components.ui_helpers import render_header, badge, status_row


def render():
    render_header()

    st.markdown("# System Status")

    # ── Component Status ──────────────────────────────────
    st.markdown("## Components")

    checks = {}

    # Check wafer model
    from inference.wafer_inference import MODEL_PATH, LABEL_MAP_PATH
    checks['Wafer Model'] = os.path.exists(MODEL_PATH)
    checks['Label Map'] = os.path.exists(LABEL_MAP_PATH)

    # Check SECOM model
    from inference.secom_inference import MODEL_PATH as SECOM_MODEL, IMPUTER_PATH, SCALER_PATH, FEATURE_COLS_PATH
    checks['Process Model'] = os.path.exists(SECOM_MODEL)
    checks['SECOM Imputer'] = os.path.exists(IMPUTER_PATH)
    checks['SECOM Scaler'] = os.path.exists(SCALER_PATH)
    checks['SECOM Features'] = os.path.exists(FEATURE_COLS_PATH)

    # Check data sources
    from inference.wafer_inference import DEMO_DATA_PATH
    checks['Wafer Demo Data'] = os.path.exists(DEMO_DATA_PATH)

    from inference.secom_inference import SECOM_DATA_PATH
    checks['SECOM Source Data'] = os.path.exists(SECOM_DATA_PATH)

    # Check SHAP availability
    try:
        import shap
        checks['SHAP Library'] = True
    except ImportError:
        checks['SHAP Library'] = False

    # Check Grad-CAM (TF)
    try:
        import tensorflow as tf
        checks['TensorFlow'] = True
    except ImportError:
        checks['TensorFlow'] = False

    # Render table
    rows_html = ""
    for name, ok in checks.items():
        if ok:
            rows_html += status_row(name, "READY", "ready")
        else:
            rows_html += status_row(name, "UNAVAILABLE", "error")

    st.markdown(f'''
    <table style="width:100%;">
        <thead><tr><th>Component</th><th>Status</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    ''', unsafe_allow_html=True)

    st.markdown("---")

    # ── Environment ───────────────────────────────────────
    st.markdown("## Environment")

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(f"**Mode:** {badge('OFFLINE ANALYTICS', 'neutral')}", unsafe_allow_html=True)
        st.markdown(f"**Data Sources:** {badge('AVAILABLE', 'ready') if all(checks.values()) else badge('PARTIAL', 'warning')}", unsafe_allow_html=True)

    with ec2:
        import sys
        st.markdown(f"**Python:** `{sys.version.split()[0]}`")
        try:
            import tensorflow as tf
            st.markdown(f"**TensorFlow:** `{tf.__version__}`")
        except:
            st.markdown("**TensorFlow:** Unavailable")
        try:
            import xgboost as xgb
            st.markdown(f"**XGBoost:** `{xgb.__version__}`")
        except:
            st.markdown("**XGBoost:** Unavailable")

    # ── Diagnostics ───────────────────────────────────────
    with st.expander("Developer Diagnostics"):
        for name, ok in checks.items():
            icon = "✅" if ok else "❌"
            st.text(f"{icon} {name}")

        st.text(f"\nWorking directory: {os.getcwd()}")
        st.text(f"Wafer model path: {MODEL_PATH}")
        st.text(f"SECOM model path: {SECOM_MODEL}")
