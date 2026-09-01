"""UI modules and tab renderers for Streamlit application."""
from src.ui.styles import apply_custom_styles
from src.ui.presets import PRESETS, RECEPTOR_STRUCT_DB, GPCRDB_CATALOG_RECORDS
from src.ui.renderers_3d import render_3dmol_conformer, render_3dmol_complex
from src.ui.tab_overview import render_tab_overview
from src.ui.tab_structure import render_tab_structure
from src.ui.tab_selectivity import render_tab_selectivity
from src.ui.tab_efficacy import render_tab_efficacy
from src.ui.tab_safety import render_tab_safety
from src.ui.tab_druglikeness import render_tab_druglikeness
from src.ui.tab_neighbors import render_tab_neighbors
from src.ui.tab_shap import render_tab_shap
from src.ui.tab_structural import render_tab_structural
from src.ui.tab_provenance import render_tab_provenance
from src.ui.tab_batch import render_tab_batch
from src.ui.tab_benchmarks import render_tab_benchmarks
from src.ui.tab_gallery import render_tab_gallery

__all__ = [
    "apply_custom_styles", "PRESETS", "RECEPTOR_STRUCT_DB", "GPCRDB_CATALOG_RECORDS",
    "render_3dmol_conformer", "render_3dmol_complex", "render_tab_overview",
    "render_tab_structure", "render_tab_selectivity", "render_tab_efficacy",
    "render_tab_safety", "render_tab_druglikeness", "render_tab_neighbors",
    "render_tab_shap", "render_tab_structural", "render_tab_provenance",
    "render_tab_batch", "render_tab_benchmarks", "render_tab_gallery",
]
