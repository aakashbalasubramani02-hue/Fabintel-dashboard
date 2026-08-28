"""
FABINTEL — Defect Review Page
Gallery view of session analyses + dataset Pareto.
"""
import streamlit as st
import numpy as np
from components.ui_helpers import render_header, badge, empty_state, confidence_badge
from inference.wafer_inference import predict_wafer, generate_gradcam_images, load_demo_samples, get_class_names
import plotly.graph_objects as go


def render():
    render_header()

    st.markdown("# Defect Review")

    tab_gallery, tab_pareto = st.tabs(["Gallery", "Defect Pareto"])

    # ── Gallery Tab ───────────────────────────────────────
    with tab_gallery:
        st.markdown("### Session Analyses")

        wafer_history = [h for h in st.session_state.get('session_history', []) if h.get('type') == 'wafer']

        if len(wafer_history) == 0:
            # Show demo gallery instead
            st.caption("No session analyses yet. Showing demonstration samples.")

            demo_samples = load_demo_samples(n_per_class=1)

            # Filter controls
            fc1, fc2 = st.columns(2)
            with fc1:
                class_names = get_class_names()
                class_filter = st.selectbox("Filter by Class", ["All"] + class_names)
            with fc2:
                pass

            if class_filter != "All":
                demo_samples = [s for s in demo_samples if s['failureType'] == class_filter]

            # Gallery grid
            cols = st.columns(4)
            for i, sample in enumerate(demo_samples):
                with cols[i % 4]:
                    # Small wafer preview
                    img_rgb = np.zeros((*sample['waferMap'].shape, 3), dtype=np.uint8)
                    img_rgb[sample['waferMap'] == 1] = [128, 128, 128]
                    img_rgb[sample['waferMap'] == 2] = [255, 255, 255]
                    st.image(img_rgb, use_container_width=True)
                    st.markdown(f"**{sample['failureType']}**")
                    st.caption(f"Label: {sample['label']}")

                    if st.button(f"Inspect", key=f"demo_inspect_{i}"):
                        result = predict_wafer(sample['waferMap'])
                        gradcam = generate_gradcam_images(sample['waferMap'], result['resized'], result['input_array'], result['class_index'])
                        st.session_state['_review_detail'] = {
                            'result': result, 'gradcam': gradcam,
                            'raw_img': sample['waferMap'], 'true_label': sample['failureType'],
                        }
        else:
            # Show session analyses
            cols = st.columns(4)
            for i, h in enumerate(reversed(wafer_history)):
                with cols[i % 4]:
                    st.markdown(f"**{h.get('class', '—')}**")
                    st.caption(f"{h.get('time', '')} | Conf: {h.get('confidence', 0):.1%}")
                    conf_label, conf_type = confidence_badge(h.get('confidence', 0))
                    st.markdown(badge(conf_label, conf_type), unsafe_allow_html=True)

        # ── Detail View ───────────────────────────────────
        detail = st.session_state.get('_review_detail')
        if detail:
            st.markdown("---")
            st.markdown("### Inspection Detail")

            dc1, dc2 = st.columns([3, 2])
            with dc1:
                view = st.radio("View", ["Original", "Grad-CAM", "Overlay"], horizontal=True, key="review_view")
                if view == "Original":
                    st.image(detail['gradcam']['original'], use_container_width=True)
                elif view == "Grad-CAM":
                    st.image(detail['gradcam']['heatmap'], use_container_width=True)
                else:
                    st.image(detail['gradcam']['overlay'], use_container_width=True)

            with dc2:
                r = detail['result']
                conf_label, conf_type = confidence_badge(r['confidence'])
                st.markdown(f"**Predicted:** {r['class']}")
                if 'true_label' in detail:
                    st.markdown(f"**True label:** {detail['true_label']}")
                st.markdown(f"**Confidence:** {r['confidence']:.1%}")
                st.markdown(f"**Status:** {badge(conf_label, conf_type)}", unsafe_allow_html=True)

                # Class probabilities table
                st.markdown("**Class Probabilities**")
                for cls in sorted(r['probabilities'].keys(), key=lambda c: r['probabilities'][c], reverse=True):
                    prob = r['probabilities'][cls]
                    marker = "●" if cls == r['class'] else " "
                    st.text(f"{marker} {cls:<12} {prob:.4f}")

    # ── Pareto Tab ────────────────────────────────────────
    with tab_pareto:
        st.markdown("### Defect Pareto")
        st.caption("DATASET STATISTIC — WM-811K labeled data")

        classes = ['none', 'Edge-Ring', 'Edge-Loc', 'Center', 'Loc', 'Scratch', 'Random', 'Donut', 'Near-full']
        counts = [123571, 9680, 5189, 4298, 3593, 1194, 866, 555, 149]
        total = sum(counts)
        cumulative = []
        running = 0
        for c in counts:
            running += c
            cumulative.append(running / total * 100)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=classes, y=counts, name='Count',
            marker_color='#2a6cb6',
            text=[f'{c:,}' for c in counts],
            textposition='outside',
            textfont=dict(size=10, family='JetBrains Mono'),
        ))
        fig.add_trace(go.Scatter(
            x=classes, y=cumulative, name='Cumulative %',
            yaxis='y2', mode='lines+markers',
            line=dict(color='#c62828', width=2),
            marker=dict(size=5),
        ))
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=40, t=10, b=40),
            yaxis=dict(title='Count', gridcolor='#eef0f2'),
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 105]),
            font=dict(family='Inter', size=11, color='#3d4f5f'),
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            legend=dict(orientation='h', y=1.08),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
