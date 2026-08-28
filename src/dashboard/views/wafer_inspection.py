"""
FABINTEL — Wafer Inspection Page
Upload wafer map → classify → Grad-CAM → engineering review.
"""
import streamlit as st
import numpy as np
from datetime import datetime
from components.ui_helpers import render_header, badge, kv_pair, panel, caution, empty_state, confidence_badge
from inference.wafer_inference import predict_wafer, generate_gradcam_images, load_demo_samples, get_class_names, get_model_info
import plotly.graph_objects as go


def render():
    render_header()

    st.markdown("# Wafer Inspection")
    st.markdown("Defect classification and spatial review")

    # ── Input Bar ─────────────────────────────────────────
    col_upload, col_demo, col_clear = st.columns([3, 2, 1])

    with col_upload:
        uploaded = st.file_uploader("Upload Wafer Map (.npy)", type=['npy'], label_visibility='collapsed')

    with col_demo:
        demo_samples = load_demo_samples(n_per_class=2)
        demo_options = ["— Select Sample —"] + [f"{s['failureType']} (#{i})" for i, s in enumerate(demo_samples)]
        demo_choice = st.selectbox("Select Sample", demo_options, label_visibility='collapsed')

    with col_clear:
        if st.button("Clear"):
            st.session_state.current_wafer_result = None
            st.rerun()

    # ── Determine input ───────────────────────────────────
    raw_img = None
    input_desc = "—"

    if uploaded is not None:
        try:
            raw_img = np.load(uploaded)
            input_desc = f"Uploaded: {uploaded.name}"
        except Exception as e:
            st.error(f"Invalid file: {e}")

    elif demo_choice != "— Select Sample —":
        idx = demo_options.index(demo_choice) - 1
        raw_img = demo_samples[idx]['waferMap']
        input_desc = f"Demo: {demo_samples[idx]['failureType']}"

    # ── Run inference ─────────────────────────────────────
    if raw_img is not None:
        with st.spinner("Analyzing..."):
            result = predict_wafer(raw_img)
            gradcam = generate_gradcam_images(raw_img, result['resized'], result['input_array'], result['class_index'])

        st.session_state.current_wafer_result = {
            'result': result,
            'gradcam': gradcam,
            'raw_img': raw_img,
            'input_desc': input_desc,
            'time': datetime.now().strftime("%H:%M:%S"),
        }

        # Add to session history
        conf_label, conf_type = confidence_badge(result['confidence'])
        st.session_state.session_history.append({
            'type': 'wafer',
            'time': datetime.now().strftime("%H:%M:%S"),
            'input_desc': input_desc,
            'class': result['class'],
            'confidence': result['confidence'],
            'status': conf_label,
        })

    # ── Display Results ───────────────────────────────────
    wr = st.session_state.get('current_wafer_result')
    if wr is None:
        empty_state("🔬", "NO WAFER SELECTED", "Upload a wafer map or select a demonstration sample to begin inspection.")
        return

    result = wr['result']
    gradcam = wr['gradcam']
    raw_img = wr['raw_img']
    info = get_model_info()

    # ── Input Status ──────────────────────────────────────
    st.markdown("### Input")
    ic1, ic2, ic3 = st.columns(3)
    ic1.markdown(f"**Original dimensions:** {raw_img.shape[0]} × {raw_img.shape[1]}")
    ic2.markdown(f"**Processed dimensions:** 128 × 128")
    ic3.markdown(f'**Status:** {badge("READY", "ready")}', unsafe_allow_html=True)

    st.markdown("---")

    # ── Main Layout: Viewer (60%) + Results (40%) ─────────
    col_viewer, col_results = st.columns([3, 2])

    with col_viewer:
        st.markdown("### Wafer Viewer")

        view_tab = st.radio("View", ["Original", "Grad-CAM", "Overlay"], horizontal=True, label_visibility='collapsed')

        if view_tab == "Original":
            st.image(gradcam['original'], caption="Original Wafer Map", use_container_width=True)
        elif view_tab == "Grad-CAM":
            st.image(gradcam['heatmap'], caption="Grad-CAM Attention Map", use_container_width=True)
        else:
            st.image(gradcam['overlay'], caption="Grad-CAM Overlay", use_container_width=True)

    with col_results:
        st.markdown("### Classification Result")

        conf_label, conf_type = confidence_badge(result['confidence'])

        html = f'''<div class="fi-panel">
            <div class="fi-panel-header">PREDICTION</div>
            {kv_pair("Defect Class", result['class'])}
            {kv_pair("Confidence", f"{result['confidence']:.1%}")}
            <div class="fi-kv-label">Status</div>
            <div class="fi-kv-value">{badge(conf_label, conf_type)}</div>
            {kv_pair("Model", info['name'])}
            {kv_pair("Input", info['input_resolution'])}
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

        # ── Class Probabilities ───────────────────────────
        st.markdown("### Class Probabilities")
        class_names = get_class_names()
        probs = result['probabilities']

        fig = go.Figure()
        sorted_classes = sorted(probs.keys(), key=lambda x: probs[x], reverse=True)
        fig.add_trace(go.Bar(
            y=sorted_classes,
            x=[probs[c] for c in sorted_classes],
            orientation='h',
            marker_color=['#2a6cb6' if c == result['class'] else '#c4cad2' for c in sorted_classes],
            text=[f'{probs[c]:.3f}' for c in sorted_classes],
            textposition='outside',
            textfont=dict(size=10, family='JetBrains Mono'),
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=5, r=40, t=5, b=5),
            xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor='#eef0f2'),
            yaxis=dict(autorange='reversed'),
            font=dict(family='Inter', size=11, color='#3d4f5f'),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Visual Evidence ───────────────────────────────────
    st.markdown("## Visual Evidence")
    ev1, ev2, ev3 = st.columns(3)
    with ev1:
        st.markdown("**ORIGINAL**")
        st.image(gradcam['original'], use_container_width=True)
    with ev2:
        st.markdown("**GRAD-CAM**")
        st.image(gradcam['heatmap'], use_container_width=True)
    with ev3:
        st.markdown("**OVERLAY**")
        st.image(gradcam['overlay'], use_container_width=True)

    # ── Inspection Summary ────────────────────────────────
    st.markdown("## Inspection Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**Predicted defect:** {result['class']}")
        st.markdown(f"**Confidence:** {result['confidence']:.1%}")
        st.markdown(f"**Model:** {info['name']}")
        st.markdown(f"**Input resolution:** {info['input_resolution']}")
    with s2:
        st.markdown(f"**Defect evidence:** Grad-CAM available")
        st.markdown(f"**Artifact:** `{info['artifact']}`")

    st.markdown(caution(
        "MODEL LIMITATION — Grad-CAM indicates model-associated spatial attention. "
        "It does not establish physical causality."
    ), unsafe_allow_html=True)
