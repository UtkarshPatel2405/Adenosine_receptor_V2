import json
import logging
import requests
from pathlib import Path
from src.pdb_utils import pdb_to_smiles

logger = logging.getLogger(__name__)

UNIPROT_MAP = {
    "A1": "P30542",
    "A2A": "P29274",
    "A2B": "P29275",
    "A3": "P33765"
}

def build_pdb_ligand_registry():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting building of Adenosine receptor PDB ligand registry...")
    
    registry = {}
    
    for subtype, uniprot_id in UNIPROT_MAP.items():
        logger.info(f"Querying RCSB PDB for subtype {subtype} (UniProt: {uniprot_id})...")
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": 100
                }
            }
        }
        
        try:
            r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=20)
            r.raise_for_status()
            result_set = r.json().get("result_set", [])
            pdb_ids = [x["identifier"] for x in result_set]
            logger.info(f"Found {len(pdb_ids)} PDB IDs for subtype {subtype}.")
            
            registry[subtype] = []
            
            for pdb_id in pdb_ids:
                logger.info(f"  Fetching ligands for PDB ID {pdb_id}...")
                ligands = pdb_to_smiles(pdb_id)
                # Filter out standard crystallization helpers/non-ligands
                real_ligands = []
                for lig in ligands:
                    # Ignore common crystallization additives if not already excluded
                    if lig["ccd"] in ("HOH", "DOD", "GOL", "SO4", "PO4", "NA", "CL", "EDT", "ACT", "DMS", "PEG", "EDO", "FMT", "BU1"):
                        continue
                    if lig["smiles"]:
                        real_ligands.append(lig)
                
                if real_ligands:
                    registry[subtype].append({
                        "pdb_id": pdb_id,
                        "ligands": real_ligands
                    })
                    
        except Exception as e:
            logger.error(f"Error building registry for subtype {subtype}: {e}")
            
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adenosine_pdb_ligands.json"
    
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)
        
    logger.info(f"Saved PDB ligand registry to {out_path}")

if __name__ == "__main__":
    build_pdb_ligand_registry()
