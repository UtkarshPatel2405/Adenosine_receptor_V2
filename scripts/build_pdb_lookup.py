from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
_TIMEOUT = 10
_BATCH_SIZE = 50
_OUTPUT = ROOT / "data" / "processed" / "smiles_to_pdb.json"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "AR-PDB-Lookup-Builder/1.0"


def _post(payload: dict) -> dict | None:
    try:
        r = _SESSION.post(RCSB_SEARCH, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def smiles_hash(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:12]


def query_pdb_for_smiles(smiles: str) -> list[dict]:
    query = {
        "query": {
            "type": "terminal",
            "service": "chemical",
            "parameters": {"value": smiles, "type": "exact_match"},
        },
        "return_type": "chemical",
        "request_options": {"paginate": {"start": 0, "rows": 10}},
    }
    result = _post(query)
    if not result:
        return []

    ligands_found = []
    for r in result.get("result_set") or []:
        ccd = r.get("identifier", "")
        if ccd:
            try:
                resp = _SESSION.get(
                    f"https://data.rcsb.org/rest/v1/core/chemical/{ccd}",
                    timeout=_TIMEOUT,
                )
                if resp.status_code == 200:
                    chem = resp.json()
                    sm = (
                        chem.get("rcsb_chem_comp_descriptors", {})
                        .get("SMILES", {})
                        .get("value", "")
                    )
                    if sm and sm.replace(" ", "") == smiles.replace(" ", ""):
                        # Find PDB entries containing this ligand
                        pdb_query = {
                            "query": {
                                "type": "terminal",
                                "service": "structure",
                                "parameters": {
                                    "value": ccd,
                                    "attribute": "rcsb_comp_id",
                                },
                            },
                            "return_type": "entry",
                            "request_options": {"paginate": {"start": 0, "rows": 5}},
                        }
                        pdb_result = _post(pdb_query)
                        for pdb_r in (pdb_result.get("result_set") or []) if pdb_result else []:
                            pdb_id = pdb_r.get("identifier", "")
                            if pdb_id:
                                ligands_found.append({
                                    "pdb_id": pdb_id,
                                    "ligand_ccd": ccd,
                                    "name": chem.get("name", ccd),
                                })
            except Exception:
                continue
    return ligands_found


def main():
    print(f"Loading training SMILES from {ROOT / 'data/processed/train_smiles.pkl'}...")
    import pickle
    try:
        with open(ROOT / "data/processed/train_smiles.pkl", "rb") as f:
            train_smiles = pickle.load(f)
    except FileNotFoundError:
        print("Training SMILES not found, using db_lookup_train.json keys...")
        with open(ROOT / "data/processed/db_lookup_train.json") as f:
            db = json.load(f)
        train_smiles = list(db.keys())

    print(f"Total molecules: {len(train_smiles)}")

    if _OUTPUT.exists():
        with open(_OUTPUT) as f:
            existing = json.load(f)
    else:
        existing = {}

    total = len(train_smiles)
    done = len(existing)
    hits = sum(1 for v in existing.values() if v)

    for i, smi in enumerate(train_smiles):
        h = smiles_hash(smi)
        if h in existing:
            continue
        result = query_pdb_for_smiles(smi)
        if result:
            existing[h] = result
            hits += 1
        else:
            existing[h] = []
        done += 1

        if done % _BATCH_SIZE == 0 or done == total:
            with open(_OUTPUT, "w") as f:
                json.dump(existing, f, indent=1)
            print(
                f"  Progress: {done}/{total} | Hits: {hits} | No hit: {done - hits}"
            )
        time.sleep(0.3)

    with open(_OUTPUT, "w") as f:
        json.dump(existing, f, indent=1)
    print(f"\nDone! {hits}/{total} molecules matched to PDB entries.")
    print(f"Output: {_OUTPUT}")


if __name__ == "__main__":
    main()
