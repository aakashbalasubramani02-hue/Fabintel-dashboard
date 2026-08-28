"""
FABINTEL — Overview Page
Engineering workspace landing: session summary, dataset statistics.
"""
import streamlit as st
from components.ui_helpers import render_header, badge, panel, kv_pair, empty_state, divider
import plotly.graph_objects as go


def render():
    render_header()

    st.markdown("# Analysis Overview")
    st.markdown("Wafer inspection and process-risk analysis")

    # ── Current Session Summary ───────────────────────────
    st.markdown("## Current Session")

    history = st.session_state.get('session_history', [])
    wafer_analyses = sum(1 for h in history if h.get('type') == 'wafer')
    process_analyses = sum(1 for h in history if h.get('type') == 'process')
    defects_detected = sum(1 for h in history if h.get('type') == 'wafer' and h.get('class', 'none') != 'none')
    high_risk = sum(1 for h in history if h.get('type') == 'process' and h.get('risk') == 'HIGH')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wafer Analyses", wafer_analyses if wafer_analyses > 0 else "—")
    c2.metric("Process Analyses", process_analyses if process_analyses > 0 else "—")
    c3.metric("Defects Detected", defects_detected if defects_detected > 0 else "—")
    c4.metric("High-Risk Analyses", high_risk if high_risk > 0 else "—")

    # ── Defect Distribution (Dataset Statistic) ───────────
    st.markdown("## Defect Distribution")
    st.caption("DATASET STATISTIC — WM-811K labeled training data")

    # These are the actual dataset class counts from Phase 1 preprocessing
    classes = ['none', 'Edge-Ring', 'Edge-Loc', 'Center', 'Loc', 'Scratch', 'Random', 'Donut', 'Near-full']
    counts = [123571, 9680, 5189, 4298, 3593, 1194, 866, 555, 149]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=classes, x=counts,
        orientation='h',
        marker_color='#2a6cb6',
        text=[f'{c:,}' for c in counts],
        textposition='outside',
        textfont=dict(size=11, family='Inter'),
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Count",
        yaxis=dict(autorange='reversed'),
        font=dict(family='Inter', size=12, color='#3d4f5f'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        xaxis=dict(gridcolor='#eef0f2', showline=True, linecolor='#dde0e4'),
        yaxis_gridcolor='#eef0f2',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Session Activity ──────────────────────────────────
    st.markdown("## Session Activity")

    if len(history) == 0:
        empty_state("📋", "NO ANALYSES IN CURRENT SESSION", "Upload a wafer map or process data to begin.")
    else:
        import pandas as pd
        rows = []
        for h in reversed(history):
            rows.append({
                'Time': h.get('time', '—'),
                'Analysis': h.get('type', '—').upper(),
                'Input': h.get('input_desc', '—'),
                'Result': h.get('class', h.get('prediction', '—')),
                'Confidence': f"{h.get('confidence', 0):.1%}" if h.get('confidence') else '—',
                'Status': h.get('status', '—'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
