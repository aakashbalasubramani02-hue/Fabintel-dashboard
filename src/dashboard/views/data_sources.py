"""
FABINTEL — Data Sources Page
Documents available datasets and their relationship.
"""
import streamlit as st
from components.ui_helpers import render_header, badge, kv_pair, panel, caution


def render():
    render_header()

    st.markdown("# Data Sources")

    # ── WM-811K ───────────────────────────────────────────
    st.markdown("## WM-811K")

    d1, d2 = st.columns(2)
    with d1:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">DATASET</div>
            {kv_pair("Purpose", "Wafer defect classification")}
            {kv_pair("Data Type", "Spatial wafer maps (2D arrays)")}
            {kv_pair("Total Labeled Samples", "172,950")}
            {kv_pair("Defect Classes", "9")}
            {kv_pair("Source", "WM-811K benchmark dataset")}
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    with d2:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">SPLIT CONFIGURATION</div>
            {kv_pair("Training", "43,484 samples")}
            {kv_pair("Validation", "10,871 samples")}
            {kv_pair("Test (CLOSED)", "118,595 samples")}
            <div class="fi-kv-label">Status</div>
            <div class="fi-kv-value">{badge("AVAILABLE", "ready")}</div>
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ── SECOM ─────────────────────────────────────────────
    st.markdown("## SECOM")

    s1, s2 = st.columns(2)
    with s1:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">DATASET</div>
            {kv_pair("Purpose", "Process failure analysis")}
            {kv_pair("Data Type", "Process measurements (continuous)")}
            {kv_pair("Total Samples", "1,567")}
            {kv_pair("Features", "590 (434 after filtering)")}
            {kv_pair("Pass / Fail", "1,463 / 104")}
            {kv_pair("Source", "UCI SECOM dataset")}
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    with s2:
        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">SPLIT CONFIGURATION</div>
            {kv_pair("Split Strategy", "Chronological 70 / 15 / 15")}
            {kv_pair("Training", "1,096 samples (78 Fail)")}
            {kv_pair("Validation", "235 samples (17 Fail)")}
            {kv_pair("Test", "236 samples (9 Fail)")}
            <div class="fi-kv-label">Status</div>
            <div class="fi-kv-value">{badge("AVAILABLE", "ready")}</div>
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Data Relationship ─────────────────────────────────
    st.markdown("## Data Relationship")

    st.info(
        "**WM-811K** and **SECOM** are complementary datasets and do not have "
        "a shared sample-level identifier. Wafer maps (WM-811K) cannot be joined "
        "to process measurement records (SECOM) at the individual wafer level."
    )

    st.markdown("""
    | Property | WM-811K | SECOM |
    |----------|---------|-------|
    | Sample identifiers | `lotName`, `waferIndex` | Timestamp only |
    | Data modality | Spatial image (2D array) | Tabular (continuous sensors) |
    | Target | Defect class (9 classes) | Binary pass/fail |
    | Linkage key | None | None |
    """)

    st.markdown(caution(
        "These datasets are used as independent modules within FABINTEL. "
        "No causal linkage between SECOM features and WM-811K defect classes is claimed."
    ), unsafe_allow_html=True)
