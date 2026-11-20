"""Fetch full-range curated human adenosine receptor bioactivities from ChEMBL REST API."""

import csv
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

TARGETS: Dict[str, str] = {
    "A1": "CHEMBL226",
    "A2A": "CHEMBL251",
    "A2B": "CHEMBL255",
    "A3": "CHEMBL256",
}

OUT_CSV = Path("data/raw/chembl_human_ar_bioactivities.csv")
FIELDNAMES = [
    "molecule_chembl_id",
    "smiles",
    "target_subtype",
    "target_chembl_id",
    "pchembl_value",
    "standard_type",
    "standard_relation",
    "assay_type",
    "assay_chembl_id",
    "document_year",
    "standard_value",
    "standard_units",
]


def _get_json(url: str, max_retries: int = 6, timeout: int = 60) -> Dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "AdenosineReceptorProfiler/2.0"}
    last_err: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504):
                sleep_s = min(30.0, (2 ** attempt) * 0.8)
                logger.warning("HTTP %s on %s; retrying in %.1fs", exc.code, url, sleep_s)
                time.sleep(sleep_s)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            sleep_s = min(30.0, (2 ** attempt) * 0.8)
            logger.warning("Network/parse error (%s) on %s; retrying in %.1fs", exc, url, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed ChEMBL API request after {max_retries} attempts: {url}") from last_err


def verify_target(subtype: str, target_id: str) -> str:
    url = f"{CHEMBL_BASE}/target/{target_id}.json"
    data = _get_json(url)
    organism = data.get("organism", "")
    pref_name = data.get("pref_name", "")
    target_type = data.get("target_type", "")
    if organism != "Homo sapiens":
        raise RuntimeError(f"Target {target_id} ({subtype}) organism mismatch: {organism!r}")
    logger.info("Verified %s -> %s: %s (%s, %s)", subtype, target_id, pref_name, organism, target_type)
    return pref_name


def fetch_target_activities(subtype: str, target_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    total_reported = None

    base_params = {
        "target_chembl_id": target_id,
        "standard_type__in": "Ki,Kd,IC50,EC50,AC50",
        "standard_relation": "=",
        "pchembl_value__isnull": "false",
        "assay_type__in": "B,F",
        "limit": str(limit),
    }

    while True:
        params = dict(base_params)
        params["offset"] = str(offset)
        query = urllib.parse.urlencode(params)
        url = f"{CHEMBL_BASE}/activity.json?{query}"
        payload = _get_json(url)

        page_meta = payload.get("page_meta") or {}
        if total_reported is None:
            total_reported = page_meta.get("total_count", "?")
            logger.info("%s (%s): ChEMBL reports %s total activity records before client filters", subtype, target_id, total_reported)

        activities = payload.get("activities") or []
        if not activities:
            break

        for act in activities:
            smiles = (act.get("canonical_smiles") or "").strip()
            if not smiles:
                continue
            if act.get("data_validity_comment"):
                continue
            if act.get("potential_duplicate") not in (0, "0", False, None):
                continue
            act_org = act.get("target_organism")
            if act_org and act_org != "Homo sapiens":
                continue

            raw_pchembl = act.get("pchembl_value")
            try:
                pchembl = float(raw_pchembl)
            except (TypeError, ValueError):
                continue
            if not (3.0 <= pchembl <= 12.5):
                continue

            std_type = str(act.get("standard_type") or "").strip().upper()
            if std_type not in {"KI", "KD", "IC50", "EC50", "AC50"}:
                continue

            rows.append(
                {
                    "molecule_chembl_id": act.get("molecule_chembl_id") or "",
                    "smiles": smiles,
                    "target_subtype": subtype,
                    "target_chembl_id": target_id,
                    "pchembl_value": round(pchembl, 4),
                    "standard_type": "Ki" if std_type == "KI" else ("Kd" if std_type == "KD" else std_type),
                    "standard_relation": "=",
                    "assay_type": str(act.get("assay_type") or "").strip().upper(),
                    "assay_chembl_id": act.get("assay_chembl_id") or "",
                    "document_year": act.get("document_year") if act.get("document_year") is not None else "",
                    "standard_value": act.get("standard_value") if act.get("standard_value") is not None else "",
                    "standard_units": act.get("standard_units") or "",
                }
            )

        offset += len(activities)
        if not page_meta.get("next"):
            break
        time.sleep(0.05)

    logger.info("%s (%s): kept %d filtered records across full potency range", subtype, target_id, len(rows))
    return rows


def main(output_csv: Path = OUT_CSV) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    for subtype, tid in TARGETS.items():
        verify_target(subtype, tid)
        all_rows.extend(fetch_target_activities(subtype, tid))

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    n_lt6 = sum(1 for r in all_rows if float(r["pchembl_value"]) < 6.0)
    logger.info(
        "Wrote %d total ChEMBL rows (%d with pChEMBL < 6.0) to %s",
        len(all_rows),
        n_lt6,
        output_csv,
    )
    return output_csv


if __name__ == "__main__":
    main()
