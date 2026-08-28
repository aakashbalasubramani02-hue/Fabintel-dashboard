"""
FABINTEL — Semiconductor Defect & Process Intelligence Platform
Main application entry point.
Launch: streamlit run src/dashboard/app.py
"""
import os
import sys
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__))))

st.set_page_config(
    page_title="FABINTEL",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────
if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'current_wafer_result' not in st.session_state:
    st.session_state.current_wafer_result = None
if 'current_process_result' not in st.session_state:
    st.session_state.current_process_result = None

# ── Sidebar Navigation ───────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ FABINTEL")
    st.caption("Semiconductor Defect &\nProcess Intelligence")

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#7f8c9b; margin-bottom:0.3rem;">WORKSPACE</p>', unsafe_allow_html=True)

    nav_options = [
        "Overview",
        "Wafer Inspection",
        "Defect Review",
        "Process Analytics",
        "Risk Investigation",
        "Reports",
    ]

    system_options = [
        "Model Registry",
        "Data Sources",
        "System Status",
    ]

    all_options = nav_options + ["---"] + system_options

    # Use a single selectbox for clean routing
    page = st.radio(
        "Navigation",
        nav_options,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#7f8c9b; margin-bottom:0.3rem;">SYSTEM</p>', unsafe_allow_html=True)

    system_page = st.radio(
        "System Navigation",
        system_options,
        index=None,
        label_visibility="collapsed",
    )

# ── Page Router ───────────────────────────────────────────
from views import overview, wafer_inspection, defect_review, process_analytics, risk_investigation, reports, model_registry, data_sources, system_status


page_map = {
    "Overview": overview,
    "Wafer Inspection": wafer_inspection,
    "Defect Review": defect_review,
    "Process Analytics": process_analytics,
    "Risk Investigation": risk_investigation,
    "Reports": reports,
    "Model Registry": model_registry,
    "Data Sources": data_sources,
    "System Status": system_status,
}

# System page takes priority if selected
active = system_page if system_page else page

if active in page_map:
    page_map[active].render()
