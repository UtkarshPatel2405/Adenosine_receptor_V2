"""Robust 3Dmol.js HTML generator for conformers and RCSB/GPCRdb complexes."""
from typing import Optional


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


def render_3dmol_complex(pdb_id: Optional[str]) -> str:
    """Generate robust HTTPS-fetched 3Dmol viewer HTML for genuine RCSB/GPCRdb PDB complex."""
    if not pdb_id or pdb_id.upper() in ("NONE", "N/A", "", "—"):
        return """<div style="background:rgba(30,41,59,0.7);border:1px dashed rgba(245,158,11,0.5);border-radius:8px;padding:2rem 1.5rem;text-align:center;color:#fcd34d;font-family:sans-serif;height:380px;display:flex;flex-direction:column;justify-content:center;align-items:center;">
            <div style="font-size:1.6rem;margin-bottom:0.5rem">🔬</div>
            <div style="font-weight:700;font-size:1rem;color:#fcd34d;margin-bottom:0.3rem">No Experimental Inactive PDB Structure Available</div>
            <div style="font-size:0.8rem;color:#94a3b8;max-width:340px;line-height:1.4;">Human A2B has <b>no deposited inactive experimental crystal/cryo-EM structure</b>. Structural models rely on AlphaFold DB (AF-P29275).</div>
            <div style="margin-top:0.8rem;"><a href="https://alphafold.ebi.ac.uk/entry/P29275" target="_blank" style="background:#f59e0b;color:#0b1120;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.75rem;font-weight:700;text-decoration:none;">View AlphaFold AF-P29275</a></div>
        </div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
    <style>html,body{{margin:0;padding:0;background:#0b1120;overflow:hidden;width:100%;height:100%;font-family:sans-serif;}}
    #viewer{{width:100%;height:400px;position:relative;border-radius:8px;background:#0e1626;}}
    #loader{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#38bdf8;font-size:0.85rem;font-weight:600;text-align:center;}}</style>
    </head><body><div id="viewer"><div id="loader">Fetching PDB: {pdb_id} from RCSB HTTPS...</div></div>
    <script>
        async function loadPDB(){{
            if (typeof $3Dmol === 'undefined') {{ setTimeout(loadPDB, 50); return; }}
            const loader = document.getElementById('loader');
            const el = document.getElementById('viewer');
            try {{
                const v = $3Dmol.createViewer(el, {{ backgroundColor: '#0e1626' }});
                const res = await fetch('https://files.rcsb.org/download/{pdb_id}.pdb');
                if (res.ok) {{
                    const txt = await res.text();
                    v.addModel(txt, "pdb");
                }} else {{
                    await new Promise((resolve) => $3Dmol.download("pdb:{pdb_id}", v, {{}}, resolve));
                }}
                v.setStyle({{}}, {{ cartoon: {{ color: 'spectrum', opacity: 0.88 }} }});
                v.setStyle({{ hetflag: true }}, {{ stick: {{ colorscheme: 'purpleCarbon', radius: 0.28 }}, sphere: {{ scale: 0.25 }} }});
                v.zoomTo(); v.render();
                loader.style.display = 'none';
                window.addEventListener('resize', () => {{ v.resize(); v.render(); }});
            }} catch (err) {{
                loader.innerHTML = '<span style="color:#f87171">Failed to fetch PDB {pdb_id}</span><br><span style="font-size:0.75rem;color:#94a3b8">Network blocked or structure unavailable</span>';
            }}
        }}
        loadPDB();
    </script></body></html>"""
