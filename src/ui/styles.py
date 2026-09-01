"""CSS style definitions and UI tokens for dark glassmorphism CADD theme."""
import streamlit as st

_CSS_CONTENT = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400..700,0..1,-50..200&display=swap');
:root {
    --bg-main: #0b1120;
    --bg-card: rgba(15, 23, 42, 0.75);
    --border-card: rgba(56, 189, 248, 0.2);
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --cyan: #38bdf8;
    --purple: #c084fc;
    --green: #4ade80;
    --amber: #fbbf24;
    --red: #f87171;
}
html, body { font-family: 'Inter', sans-serif; color: var(--text-primary); }
.material-symbols-outlined { font-family: 'Material Symbols Outlined' !important; font-variation-settings: 'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
.cadd-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}
.section-title { font-size: 1.15rem; font-weight: 700; margin: 0.2rem 0 0.2rem 0; display: flex; align-items: center; gap: 0.4rem; }
.section-subtitle { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.4rem; }
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 9999px;
    padding: 0.25rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    text-decoration: none;
}
.badge-cyan { background: rgba(14, 116, 144, 0.25); border-color: rgba(56, 189, 248, 0.4); color: var(--cyan); }
.badge-purple { background: rgba(107, 33, 168, 0.25); border-color: rgba(192, 132, 252, 0.4); color: var(--purple); }
.badge-green { background: rgba(21, 128, 61, 0.25); border-color: rgba(74, 222, 128, 0.4); color: var(--green); }
.badge-amber { background: rgba(180, 83, 9, 0.25); border-color: rgba(251, 191, 36, 0.4); color: var(--amber); }
.badge-red { background: rgba(185, 28, 28, 0.25); border-color: rgba(248, 113, 113, 0.4); color: var(--red); }
.kpi-box {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.kpi-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.kpi-value { font-size: 1.15rem; font-weight: 700; margin-top: 0.2rem; }
.theory-callout {
    background: rgba(30, 41, 59, 0.5);
    border-left: 3px solid var(--cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    margin-top: 1rem;
    font-size: 0.8rem;
    color: #cbd5e1;
    line-height: 1.5;
}
.theory-callout h4 { margin: 0 0 0.35rem 0; font-size: 0.85rem; color: var(--cyan); font-weight: 700; }
.cadd-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.cadd-table th, .cadd-table td { padding: 0.55rem 0.8rem; border-bottom: 1px solid rgba(56, 189, 248, 0.12); text-align: left; }
.cadd-table th { color: var(--cyan); font-weight: 700; background: rgba(14, 116, 144, 0.1); }
.smiles-cell { font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; word-break: break-all; color: #93c5fd; }
</style>
"""


def apply_custom_styles() -> None:
    """Inject styling block into the Streamlit application."""
    st.markdown(_CSS_CONTENT, unsafe_allow_html=True)
