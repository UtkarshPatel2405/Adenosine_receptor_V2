"""CSS style definitions and UI tokens for dark glassmorphism CADD theme."""
import streamlit as st

_CSS_CONTENT = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400..700,0..1,-50..200&display=swap');

:root {
    --bg-canvas: #1e293b;
    --bg-card: rgba(30, 41, 59, 0.85);
    --bg-card-hover: rgba(51, 65, 85, 0.9);
    --border-subtle: rgba(216, 224, 230, 0.16);
    --border-glow: rgba(56, 189, 248, 0.45);

    --cyan: #38bdf8;
    --cyan-glow: rgba(56, 189, 248, 0.25);
    --purple: #a78bfa;
    --purple-glow: rgba(167, 139, 250, 0.25);
    --green: #4ade80;
    --green-glow: rgba(74, 222, 128, 0.25);
    --red: #f87171;
    --amber: #fbbf24;

    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Multi-layer radial and linear gradient background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 50% 50%, rgba(38, 70, 83, 0.45), transparent 80%),
        radial-gradient(ellipse at 20% 80%, rgba(42, 157, 143, 0.2), transparent 70%),
        radial-gradient(ellipse at 80% 20%, rgba(38, 70, 83, 0.25), transparent 70%),
        linear-gradient(135deg, #0f172a, #182B32, #0b1120) !important;
    background-blend-mode: multiply, screen, normal, normal;
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid var(--border-subtle);
    backdrop-filter: blur(16px);
}

header[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(12px) !important;
}

/* Custom Sleek Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.7);
}
::-webkit-scrollbar-thumb {
    background: rgba(56, 189, 248, 0.35);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(56, 189, 248, 0.7);
}

/* Custom Card Container with Glow and Lift */
.cadd-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(8px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.cadd-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 30px -4px rgba(56, 189, 248, 0.2);
    transform: translateY(-2px);
}

/* Section Header styling */
.section-num {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    background: rgba(56, 189, 248, 0.12);
    color: var(--cyan);
    margin-bottom: 0.35rem;
}

.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.page-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
}

.section-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

@keyframes hit-pulse {
    0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}

.target-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}

.target-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5);
    border-color: var(--cyan);
}

.target-card.primary-hit {
    border-color: rgba(74, 222, 128, 0.6);
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(21, 128, 61, 0.15));
    animation: hit-pulse 2.5s infinite;
}

/* Potency Progress Sparkline Bar */
.potency-bar-track {
    width: 100%;
    height: 6px;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 9999px;
    margin: 0.5rem 0 0.3rem;
    overflow: hidden;
}

.potency-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.badge-cyan { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.35); }
.badge-purple { background: rgba(167, 139, 250, 0.15); color: #c4b5fd; border: 1px solid rgba(167, 139, 250, 0.35); }
.badge-green { background: rgba(74, 222, 128, 0.15); color: #86efac; border: 1px solid rgba(74, 222, 128, 0.35); }
.badge-red { background: rgba(248, 113, 113, 0.15); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.35); }
.badge-amber { background: rgba(251, 191, 36, 0.15); color: #fcd34d; border: 1px solid rgba(251, 191, 36, 0.35); }

.badge-pill:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
}

/* KPI Metric Box */
.kpi-box {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.kpi-box:hover {
    transform: translateY(-2px);
    border-color: var(--cyan);
    box-shadow: 0 4px 20px -2px rgba(56, 189, 248, 0.15);
}

.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    font-weight: 600;
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* Hero Trust & Precision Metric Strip */
.hero-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin: 1rem 0 1.25rem;
}
.hero-chip {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    flex: 1 1 auto;
    min-width: 120px;
    text-align: center;
}
.hero-chip .chip-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}
.hero-chip .chip-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* Subtype Metric Chips */
.subtype-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin-top: 0.6rem;
}
.subtype-metrics .st-chip {
    background: rgba(56, 189, 248, 0.07);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    text-align: center;
}
.subtype-metrics .st-name {
    font-weight: 700;
    color: #7dd3fc;
    font-size: 0.85rem;
}
.subtype-metrics .st-metric {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* Theory Callout */
.theory-callout {
    background: rgba(56, 189, 248, 0.06);
    border-left: 3px solid var(--cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    line-height: 1.55;
    color: var(--text-secondary);
}

.theory-callout h4 {
    color: var(--cyan) !important;
    font-size: 0.92rem !important;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* CADD Table styling */
.cadd-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.82rem;
}

.cadd-table th {
    background: rgba(30, 41, 59, 0.9);
    color: var(--text-primary);
    font-weight: 600;
    padding: 0.6rem 0.8rem;
    border-bottom: 2px solid rgba(216, 224, 230, 0.2);
    text-align: left;
}

.cadd-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid rgba(216, 224, 230, 0.1);
    color: var(--text-secondary);
}

.cadd-table tr:hover td {
    background: rgba(56, 189, 248, 0.06);
}

.smiles-cell {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    word-break: break-all;
    color: #93c5fd;
}

/* Streamlit Button Glow & Polish */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 14px var(--cyan-glow) !important;
}

.material-symbols-outlined {
    font-variation-settings: 'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 24;
    vertical-align: middle;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject styling block into the Streamlit application."""
    st.markdown(_CSS_CONTENT, unsafe_allow_html=True)

