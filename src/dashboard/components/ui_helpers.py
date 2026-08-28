"""
FABINTEL — Reusable UI Components
Status badges, metric panels, key-value displays.
"""
import streamlit as st

def badge(label, status):
    """Render a status badge. status: ready/pass/low/review/medium/warning/fail/high/error/neutral"""
    css_class = f"fi-badge fi-badge-{status.lower()}"
    return f'<span class="{css_class}">{label}</span>'

def panel_header(title):
    """Render a panel section header."""
    return f'<div class="fi-panel-header">{title}</div>'

def kv_pair(label, value):
    """Render a key-value pair."""
    return f'<div class="fi-kv-label">{label}</div><div class="fi-kv-value">{value}</div>'

def panel(title, content_html):
    """Render a bordered panel with header."""
    return f'''<div class="fi-panel">
        <div class="fi-panel-header">{title}</div>
        {content_html}
    </div>'''

def caution(text):
    """Render a caution/disclaimer notice."""
    return f'<div class="fi-caution">⚠ {text}</div>'

def divider():
    """Render a thin horizontal divider."""
    return '<div class="fi-divider"></div>'

def status_row(label, status_text, status_type='ready'):
    """Render a component-status row for system status page."""
    return f'''<tr>
        <td style="font-weight:500;">{label}</td>
        <td>{badge(status_text, status_type)}</td>
    </tr>'''

def render_header():
    """Render the FABINTEL application header."""
    st.markdown('''
    <div class="fi-header">
        <div>
            <div class="fi-header-title">FABINTEL</div>
            <div class="fi-header-sub">Semiconductor Defect & Process Intelligence</div>
        </div>
        <div>
            <span class="fi-badge fi-badge-neutral">OFFLINE ANALYTICS</span>
            &nbsp;
            <span class="fi-badge fi-badge-ready">READY</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def empty_state(icon, title, description):
    """Render a professional empty state."""
    st.markdown(f'''
    <div style="text-align:center; padding: 3rem 1rem; color: #7f8c9b;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-family: Inter, sans-serif; font-size: 0.95rem; font-weight: 600; color: #3d4f5f;">{title}</div>
        <div style="font-family: Inter, sans-serif; font-size: 0.82rem; margin-top: 0.3rem;">{description}</div>
    </div>
    ''', unsafe_allow_html=True)

def confidence_badge(conf):
    """Return appropriate status based on confidence."""
    if conf >= 0.8:
        return 'PASS', 'pass'
    elif conf >= 0.5:
        return 'REVIEW', 'review'
    else:
        return 'LOW CONFIDENCE', 'warning'

def risk_badge(risk):
    """Return badge for risk level."""
    mapping = {'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'}
    return badge(risk, mapping.get(risk, 'neutral'))
