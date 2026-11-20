"""FastAPI server and API gateway for the Adenosine Receptor Profiler."""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analysis import receptor_neighbors, shap_analysis
from src.chem_utils import (
    canonicalize,
    check_pains,
    draw_2d_svg,
    generate_pdb_block,
    generate_sdf_block,
    qed_profile,
)
from src.config import APP_VERSION, OUTPUTS_DIR, PROCESSED_DATA_DIR, SUBTYPES
from src.predictor import predict
from src.presets import GPCRDB_CATALOG_RECORDS, PRESETS, RECEPTOR_STRUCT_DB
from src.provenance import provenance_payload

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"

app = FastAPI(
    title="Adenosine Receptor Profiler API",
    version=APP_VERSION,
    description="Conformal multi-target pKi prediction and selectivity profiling for human A1, A2A, A2B, and A3 receptors.",
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

_DOWNLOAD_WHITELIST: Dict[str, Path] = {
    "A1_pki.csv": PROCESSED_DATA_DIR / "A1_pki.csv",
    "A2A_pki.csv": PROCESSED_DATA_DIR / "A2A_pki.csv",
    "A2B_pki.csv": PROCESSED_DATA_DIR / "A2B_pki.csv",
    "A3_pki.csv": PROCESSED_DATA_DIR / "A3_pki.csv",
    "evaluation_report.json": OUTPUTS_DIR / "evaluation_report.json",
    "holdout_predictions.csv": OUTPUTS_DIR / "holdout_predictions.csv",
    "dataset_diagnostics.json": OUTPUTS_DIR / "dataset_diagnostics.json",
    "training_summary.json": OUTPUTS_DIR / "training_summary.json",
    "global_split.json": PROCESSED_DATA_DIR / "global_split.json",
}


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="Input SMILES string")
    threshold: float = Field(6.0, ge=3.0, le=11.0, description="Hit cutoff pKi threshold")


class BatchRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings (up to 250)")
    threshold: float = Field(6.0, ge=3.0, le=11.0, description="Hit cutoff pKi threshold")


class Export3DRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="Input SMILES string")
    fmt: str = Field("sdf", description="Output 3D format: 'sdf' or 'pdb'")


def _normalize_qed_profile(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure consistent physicochemical, Lipinski Ro5, and Veber rule fields."""
    if not raw:
        return {}
    qed = raw.get("qed", raw.get("QED", 0.0))
    mw = raw.get("mw", raw.get("MW", 0.0))
    logp = raw.get("logp", raw.get("LogP", 0.0))
    hbd = int(raw.get("hbd", raw.get("HBD", 0)))
    hba = int(raw.get("hba", raw.get("HBA", 0)))
    tpsa = raw.get("tpsa", raw.get("TPSA", 0.0))
    rot_bonds = int(raw.get("rot_bonds", raw.get("RotB", 0)))
    arom_rings = int(raw.get("aromatic_rings", raw.get("AromaticRings", 0)))

    lip_viol = raw.get("lipinski_violations")
    if lip_viol is None:
        lip_viol = int(mw > 500.0) + int(logp > 5.0) + int(hbd > 5) + int(hba > 10)
    lip_pass = raw.get("lipinski_pass", lip_viol <= 1)

    veber_tpsa_ok = raw.get("veber_tpsa_ok", tpsa <= 140.0)
    veber_rot_ok = raw.get("veber_rotbonds_ok", rot_bonds <= 10)
    veber_viol = raw.get("veber_violations")
    if veber_viol is None:
        veber_viol = int(not veber_tpsa_ok) + int(not veber_rot_ok)
    veber_pass = raw.get("veber_pass", veber_viol == 0)

    return {
        "qed": round(float(qed), 4),
        "mw": round(float(mw), 2),
        "logp": round(float(logp), 3),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(float(tpsa), 2),
        "rot_bonds": rot_bonds,
        "aromatic_rings": arom_rings,
        "lipinski_violations": int(lip_viol),
        "lipinski_pass": bool(lip_pass),
        "veber_tpsa_ok": bool(veber_tpsa_ok),
        "veber_rotbonds_ok": bool(veber_rot_ok),
        "veber_violations": int(veber_viol),
        "veber_pass": bool(veber_pass),
    }


def _safe_read_json(path: Path) -> Optional[Any]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed reading JSON %s: %s", path, exc)
        return None


def _safe_read_holdout_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    numeric_cols = {
        "y_true",
        "y_pred_xgb",
        "y_pred_lgb",
        "y_pred_rf",
        "y_pred_consensus",
        "ci_low",
        "ci_high",
        "std_equiv",
    }
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed: Dict[str, Any] = {}
                for k, v in row.items():
                    if k in numeric_cols:
                        try:
                            parsed[k] = float(v) if v not in ("", None) else None
                        except ValueError:
                            parsed[k] = None
                    else:
                        parsed[k] = v
                rows.append(parsed)
    except Exception as exc:
        logger.warning("Failed reading holdout CSV %s: %s", path, exc)
        return []
    return rows


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return True


@app.get("/")
def serve_index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "models": ["XGBoost", "LightGBM", "RandomForest", "Consensus"],
    }


@app.get("/api/presets")
def api_presets() -> Dict[str, Any]:
    return {
        "presets": PRESETS,
        "receptors": RECEPTOR_STRUCT_DB,
        "catalog": GPCRDB_CATALOG_RECORDS,
    }


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> Dict[str, Any]:
    canon = canonicalize(req.smiles)
    if canon is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    try:
        res = predict(canon, threshold=req.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed for %s", canon)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    svg = draw_2d_svg(canon, size=(420, 280))
    if not isinstance(svg, str):
        svg = None

    qed = _normalize_qed_profile(qed_profile(canon))
    pains = check_pains(canon)
    primary = res.get("best_target", "A2A")
    if primary not in SUBTYPES:
        primary = "A2A"

    raw_neighbors = receptor_neighbors(canon, primary, top_k=5) or []
    neighbors: List[Dict[str, Any]] = []
    for nb in raw_neighbors:
        nb_item = dict(nb)
        nb_svg = draw_2d_svg(nb_item["smiles"], size=(220, 150))
        nb_item["svg"] = nb_svg if isinstance(nb_svg, str) else None
        neighbors.append(nb_item)

    res["svg_2d"] = svg
    res["qed_profile"] = qed
    res["pains_alerts"] = pains
    res["neighbors"] = neighbors
    res["threshold"] = req.threshold
    return res


@app.post("/api/batch")
def api_batch(req: BatchRequest) -> Dict[str, Any]:
    cleaned = [s.strip() for s in req.smiles_list if isinstance(s, str) and s.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="No SMILES provided.")
    if len(cleaned) > 250:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 250 SMILES.")

    rows: List[Dict[str, Any]] = []
    invalid_count = 0
    invalid_entries: List[str] = []

    for idx, raw_smi in enumerate(cleaned, start=1):
        canon = canonicalize(raw_smi)
        if canon is None:
            invalid_count += 1
            if len(invalid_entries) < 20:
                invalid_entries.append(raw_smi)
            continue
        try:
            res = predict(canon, threshold=req.threshold)
            cons = res.get("predictions", {}).get("Consensus", {})
            ad = res.get("applicability_domain", {})
            cns = res.get("cns_admet", {})
            sel = res.get("selectivity_spectrum", {})
            best_t = res.get("best_target", "None (Inactive across all subtypes)")
            top_pki = max((float(cons.get(st, 0.0) or 0.0) for st in SUBTYPES), default=0.0)
            rows.append(
                {
                    "index": idx,
                    "input_smiles": raw_smi,
                    "smiles": canon,
                    "A1": cons.get("A1"),
                    "A2A": cons.get("A2A"),
                    "A2B": cons.get("A2B"),
                    "A3": cons.get("A3"),
                    "best_target": best_t,
                    "max_pki": round(top_pki, 3),
                    "selectivity_margin": sel.get("selectivity_margin", 0.0),
                    "selectivity_fold": sel.get("selectivity_fold", 1.0),
                    "hits": res.get("target_hits", []),
                    "is_hit": len(res.get("target_hits", [])) > 0,
                    "in_domain": bool(ad.get("in_domain", False)),
                    "tanimoto_max": ad.get("tanimoto_max", 0.0),
                    "cns_mpo": cns.get("cns_mpo_score", 0.0),
                    "db_status": res.get("db_status", "de_novo"),
                }
            )
        except Exception as exc:
            logger.warning("Batch item %d failed: %s", idx, exc)
            invalid_count += 1

    rows.sort(key=lambda r: (r["max_pki"] if r["max_pki"] is not None else -1.0), reverse=True)
    return {
        "total_submitted": len(cleaned),
        "valid_count": len(rows),
        "invalid_count": invalid_count,
        "invalid_samples": invalid_entries,
        "hit_count": sum(1 for r in rows if r["is_hit"]),
        "threshold": req.threshold,
        "results": rows,
    }


@app.get("/api/neighbors")
def api_neighbors(
    smiles: str = Query(..., min_length=1),
    subtype: str = Query("A2A"),
    top_k: int = Query(5, ge=1, le=20),
) -> Dict[str, Any]:
    if subtype not in SUBTYPES:
        raise HTTPException(status_code=400, detail=f"Invalid subtype. Must be one of {SUBTYPES}.")
    canon = canonicalize(smiles)
    if canon is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    raw_neighbors = receptor_neighbors(canon, subtype, top_k=top_k) or []
    enriched: List[Dict[str, Any]] = []
    for nb in raw_neighbors:
        item = dict(nb)
        svg = draw_2d_svg(item["smiles"], size=(220, 150))
        item["svg"] = svg if isinstance(svg, str) else None
        enriched.append(item)

    return {"smiles": canon, "subtype": subtype, "neighbors": enriched}


@app.get("/api/shap")
def api_shap(
    smiles: str = Query(..., min_length=1),
    subtype: str = Query("A2A"),
    top_k: int = Query(10, ge=1, le=25),
) -> Dict[str, Any]:
    if subtype not in SUBTYPES:
        raise HTTPException(status_code=400, detail=f"Invalid subtype. Must be one of {SUBTYPES}.")
    canon = canonicalize(smiles)
    if canon is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    result = shap_analysis(canon, subtype, top_k=top_k)
    if result is None:
        raise HTTPException(status_code=404, detail=f"TreeSHAP analysis unavailable for subtype {subtype}.")
    return result


@app.get("/api/benchmarks")
def api_benchmarks() -> Dict[str, Any]:
    return {
        "evaluation_report": _safe_read_json(OUTPUTS_DIR / "evaluation_report.json"),
        "dataset_diagnostics": _safe_read_json(OUTPUTS_DIR / "dataset_diagnostics.json"),
        "training_summary": _safe_read_json(OUTPUTS_DIR / "training_summary.json"),
        "shap_summary": _safe_read_json(OUTPUTS_DIR / "shap" / "shap_summary.json"),
        "y_randomization_summary": _safe_read_json(OUTPUTS_DIR / "y_randomization" / "y_randomization_summary.json"),
        "holdout_predictions": _safe_read_holdout_csv(OUTPUTS_DIR / "holdout_predictions.csv"),
    }


@app.get("/api/provenance")
def api_provenance() -> Dict[str, Any]:
    return provenance_payload()


@app.post("/api/export/3d")
def api_export_3d(req: Export3DRequest) -> Dict[str, Any]:
    canon = canonicalize(req.smiles)
    if canon is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    fmt = req.fmt.strip().lower()
    if fmt not in ("sdf", "pdb"):
        raise HTTPException(status_code=400, detail="Format must be 'sdf' or 'pdb'.")

    content = generate_sdf_block(canon) if fmt == "sdf" else generate_pdb_block(canon)
    if not content:
        raise HTTPException(status_code=400, detail=f"Failed to generate 3D {fmt.upper()} conformer.")

    return {"format": fmt, "smiles": canon, "content": content}


@app.get("/api/download/{artifact_name}")
def api_download(artifact_name: str):
    if artifact_name not in _DOWNLOAD_WHITELIST:
        raise HTTPException(status_code=404, detail="Artifact not available for download.")

    target_path = _DOWNLOAD_WHITELIST[artifact_name]
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk.")

    size = target_path.stat().st_size
    if size <= 100 or _is_lfs_pointer(target_path):
        raise HTTPException(status_code=404, detail="Artifact is unavailable or incomplete.")

    media_type = "application/json" if artifact_name.endswith(".json") else "text/csv"
    return FileResponse(path=str(target_path), filename=artifact_name, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
