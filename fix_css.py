import sys
path = r'c:\Users\utkar\Desktop\Adenosine\Adenosine_Selectivity_Model-main\Adenosine_Selectivity_Model-main\streamlit_app.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the st.logo line which is intact.
logo_str = 'st.logo(":material/biotech:", icon_image=":material/biotech:", size="large")\n\n'
idx = content.find(logo_str)

if idx != -1:
    before = content[:idx + len(logo_str)]
    
    hover_idx = content.find('.cadd-card:hover {')
    if hover_idx != -1:
        after = content[hover_idx:]
        
        css_block = '''# ─────────────────────────────────────────────────────────────────────────────
# 1. PROFESSIONAL STYLING & MOTION TOKENS (No Generic Emojis)
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400..700,0..1,-50..200&display=swap');

:root {
    --bg-canvas: #2a3439;
    --bg-card: #333f45;
    --bg-card-hover: #3d4a52;
    --border-subtle: rgba(216, 224, 230, 0.14);
    --border-glow: rgba(56, 189, 248, 0.4);

    --cyan: #38bdf8;
    --cyan-glow: rgba(56, 189, 248, 0.2);
    --purple: #a78bfa;
    --purple-glow: rgba(167, 139, 250, 0.2);
    --green: #4ade80;
    --green-glow: rgba(74, 222, 128, 0.2);
    --red: #f87171;
    --amber: #fbbf24;

    --text-primary: #eef2f4;
    --text-secondary: #c8d0d6;
    --text-muted: #9aa7af;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    background-color: var(--bg-canvas);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle 480px at 8% 6%, rgba(56, 189, 248, 0.16), transparent 62%),
        radial-gradient(circle 420px at 94% 92%, rgba(167, 139, 250, 0.14), transparent 62%),
        radial-gradient(circle 320px at 55% 42%, rgba(74, 222, 128, 0.10), transparent 62%),
        linear-gradient(125deg, #253238, #344249, #2b393f, #38464d, #26343a) !important;
    background-size:
        480px 480px, 420px 420px, 320px 320px,
        320% 320% !important;
}

[data-testid="stSidebar"] {
    background: #333e44 !important;
    border-right: 1px solid var(--border-subtle);
}

header[data-testid="stHeader"] {
    background: rgba(42, 52, 57, 0.8) !important;
    backdrop-filter: blur(12px) !important;
}

/* Custom Card Container */
.cadd-card {
    background: rgba(51, 63, 69, 0.82);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(6px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

'''
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(before + css_block + after)
        print('Fixed CSS.')
    else:
        print('Hover not found.')
else:
    print('Logo not found.')
