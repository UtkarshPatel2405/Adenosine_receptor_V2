"""Smoke tests for FastAPI backend endpoints, static web SPA assets, and Wave 2 pure pKi benchmark integration."""
import math
from pathlib import Path

import pytest
from fastapi import HTTPException

import app as fastapi_app
from app import (
    BatchRequest,
    Export3DRequest,
    PredictRequest,
    api_batch,
    api_benchmarks,
    api_download,
    api_export_3d,
    api_health,
    api_neighbors,
    api_predict,
    api_presets,
    api_provenance,
    api_shap,
    serve_index,
)
from src.config import ROOT_DIR, SUBTYPES
from src.presets import PRESETS


class TestWebStaticAssets:
    def test_web_files_exist_and_dark_theme_configured(self):
        web_dir = Path(ROOT_DIR) / "web"
        assert (web_dir / "index.html").is_file()
        assert (web_dir / "styles.css").is_file()
        assert (web_dir / "app.js").is_file()

        css = (web_dir / "styles.css").read_text(encoding="utf-8")
        assert "#0f172a" in css
        assert "#f8fafc" in css
        assert "#38bdf8" in css

        assert not (Path(ROOT_DIR) / "streamlit_app.py").exists()
        assert not (Path(ROOT_DIR) / ".streamlit").exists()
        assert not (Path(ROOT_DIR) / "src" / "ui").exists()

    def test_serve_index_returns_html_file_response(self):
        resp = serve_index()
        assert Path(resp.path).name == "index.html"


class TestFastAPIEndpoints:
    def test_health_and_presets_endpoints(self):
        health = api_health()
        assert health["status"] == "ok"
        assert set(health["models"]) == {"XGBoost", "LightGBM", "RandomForest", "Consensus"}

        presets_payload = api_presets()
        assert len(presets_payload["presets"]) == 12
        assert set(presets_payload["receptors"].keys()) == set(SUBTYPES)
        assert len(presets_payload["catalog"]) >= 8

    def test_predict_endpoint_on_zm241385_and_invalid_smiles(self):
        zm_smiles = PRESETS["ZM241385"]["smiles"]
        res = api_predict(PredictRequest(smiles=zm_smiles, threshold=6.0))
        assert res["in_database"] is True
        assert res["db_status"] in ("training", "test")
        assert res["db_value"]["A2A"] >= 8.0

        for m_name in ("XGBoost", "LightGBM", "RandomForest", "Consensus"):
            assert m_name in res["predictions"]
            for st in SUBTYPES:
                pred = res["predictions"][m_name][st]
                iv = res["intervals"][m_name][st]
                assert math.isfinite(pred)
                assert iv["lower"] <= pred <= iv["upper"]

        assert isinstance(res["svg_2d"], str) and "<svg" in res["svg_2d"]
        assert res["qed_profile"]["veber_pass"] is True
        assert res["qed_profile"]["lipinski_pass"] is True
        assert isinstance(res["neighbors"], list)

        with pytest.raises(HTTPException) as exc:
            api_predict(PredictRequest(smiles="INVALID_SMILES_123", threshold=6.0))
        assert exc.value.status_code == 400

    def test_batch_neighbors_shap_and_3d_export_endpoints(self):
        batch_res = api_batch(
            BatchRequest(
                smiles_list=[PRESETS["ZM241385"]["smiles"], PRESETS["PSB-603"]["smiles"], "BAD_SMILES"],
                threshold=6.0,
            )
        )
        assert batch_res["total_submitted"] == 3
        assert batch_res["valid_count"] == 2
        assert batch_res["invalid_count"] == 1
        assert len(batch_res["results"]) == 2

        nb = api_neighbors(smiles=PRESETS["ZM241385"]["smiles"], subtype="A2A", top_k=3)
        assert nb["subtype"] == "A2A"
        assert len(nb["neighbors"]) > 0

        sh = api_shap(smiles=PRESETS["ZM241385"]["smiles"], subtype="A2A", top_k=5)
        assert sh["subtype"] == "A2A"
        assert len(sh["features"]) == 5

        sdf = api_export_3d(Export3DRequest(smiles="CCO", fmt="sdf"))
        assert sdf["format"] == "sdf"
        assert "V2000" in sdf["content"] or "M  END" in sdf["content"]

    def test_benchmarks_provenance_and_download_whitelist(self):
        bm = api_benchmarks()
        eval_rep = bm["evaluation_report"]
        diag = bm["dataset_diagnostics"]
        assert eval_rep is not None
        assert diag is not None
        assert eval_rep["assay_filter"] == "Ki and Kd only (pKi)"
        assert eval_rep["overall"]["spearman_rho"] > 0.70
        assert eval_rep["overall"]["r2"] > 0.50
        assert diag["standard_type_breakdown"] == {"KI": 16158, "KD": 197}
        assert len(bm["holdout_predictions"]) == 3273

        prov = api_provenance()
        assert prov["version"] == fastapi_app.APP_VERSION
        assert len(prov["artifacts"]) >= 12

        for name in ("A1_pki.csv", "A2A_pki.csv", "A2B_pki.csv", "A3_pki.csv", "evaluation_report.json"):
            dl = api_download(name)
            assert Path(dl.path).is_file()

        with pytest.raises(HTTPException) as exc:
            api_download("../../etc/passwd")
        assert exc.value.status_code == 404
