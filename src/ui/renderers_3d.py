"""Robust 3Dmol.js HTML generator for molecular conformers."""

def render_3dmol_conformer(mol_block: str) -> str:
    """Generate embedded 3Dmol viewer HTML for an isolated 3D molecular conformer."""
    clean_mol = mol_block.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
    <style>html,body{{margin:0;padding:0;background:#0b1120;overflow:hidden;width:100%;height:100%;font-family:sans-serif;}}
    #viewer{{width:100%;height:380px;position:relative;border-radius:8px;background:#0e1626;}}
    #loader{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#38bdf8;font-size:0.85rem;font-weight:600;}}</style>
    </head><body><div id="viewer"><div id="loader">Generating 3D MMFF94 Conformer...</div></div>
    <script>
        function init(){{
            if (typeof $3Dmol === 'undefined') {{ setTimeout(init, 50); return; }}
            try {{
                const el = document.getElementById('viewer');
                const v = $3Dmol.createViewer(el, {{ backgroundColor: '#0e1626' }});
                v.addModel(`{clean_mol}`, "sdf");
                v.setStyle({{}}, {{ stick: {{ colorscheme: 'cyanCarbon', radius: 0.22 }}, sphere: {{ scale: 0.26 }} }});
                v.zoomTo(); v.render();
                document.getElementById('loader').style.display = 'none';
                window.addEventListener('resize', () => {{ v.resize(); v.render(); }});
            }} catch(e) {{ document.getElementById('loader').innerText = 'Render error: ' + e.message; }}
        }}
        init();
    </script></body></html>"""
