import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from src.chem_utils import canonicalize
from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SUBTYPES, VALID_STANDARD_TYPES,
    REQUIRED_CONFIDENCE, DECOY_PCHEMBL, LOG_LEVEL,
)

logger = logging.getLogger(__name__)


SUBTYPE_MAP = {
    "A1R": "A1", "A2AR": "A2A", "A2BR": "A2B", "A3R": "A3",
    "A1": "A1", "A2A": "A2A", "A2B": "A2B", "A3": "A3",
}




def load_all_raw_data(data_dir: str | Path) -> pd.DataFrame:
    raw_path = Path(data_dir)
    dfs = []

    chembl_file = raw_path / "AR_all_unique_parents_with_smiles.csv"
    if chembl_file.exists():
        df_chembl = pd.read_csv(chembl_file)
        keep_cols = ["smiles", "pchembl_value", "standard_type", "TAG",
                      "standard_relation", "assay_type", "confidence_score", "targets_hit"]
        df_chembl = df_chembl[[c for c in keep_cols if c in df_chembl.columns]].copy()
        dfs.append(df_chembl)
        logger.info("Loaded ChEMBL CSV: %d rows.", len(df_chembl))

    for excel_file in raw_path.glob("GPCRdb_*.xlsx"):
        if excel_file.stat().st_size < 1000:
            logger.info("Skipping Git LFS pointer file %s", excel_file)
            continue
        try:
            df_gpcr = pd.read_excel(excel_file)
        except Exception as e:
            logger.warning("Could not read %s: %s. Skipping.", excel_file, e)
            continue
        tag = excel_file.stem.split("_")[1] + "R"

        rename_map = {
            "Smiles": "smiles", "p-value (-log)": "pchembl_value",
            "Activity Type": "standard_type", "Activity Relation": "standard_relation",
            "Assay Type": "assay_type",
        }
        df_gpcr = df_gpcr.rename(columns=rename_map)

        if "standard_type" in df_gpcr.columns:
            _std_type_map = {"pKi": "KI", "pKd": "KD", "pIC50": "IC50", "pEC50": "EC50", "pAC50": "AC50"}
            df_gpcr["standard_type"] = df_gpcr["standard_type"].astype(str).str.strip().replace(_std_type_map)

        df_gpcr["TAG"] = tag
        df_gpcr["confidence_score"] = 9

        if "assay_type" in df_gpcr.columns:
            df_gpcr["assay_type"] = df_gpcr["assay_type"].replace({"Binding": "B", "Functional": "F"})

        for col in ["smiles", "pchembl_value", "standard_type", "TAG",
                      "standard_relation", "assay_type", "confidence_score", "targets_hit"]:
            if col not in df_gpcr.columns:
                df_gpcr[col] = None

        df_gpcr["targets_hit"] = df_gpcr["TAG"]
        df_gpcr = df_gpcr[["smiles", "pchembl_value", "standard_type", "TAG",
                            "standard_relation", "assay_type", "confidence_score", "targets_hit"]].copy()
        dfs.append(df_gpcr)
        logger.info("Loaded GPCRdb Excel %s: %d rows.", excel_file.name, len(df_gpcr))

    if not dfs:
        raise ValueError(f"No valid datasets found in {data_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_and_clean(
    data_dir: str | Path = "data/raw",
    save_lookup_path: str | None = None,
    mode: str = "precise",
    target_role: str = "all",
    target_endpoint: str = "all",
    include_decoys: bool = False,
):
    """
    Load, filter, deduplicate, and prepare bioactivity data.

    CRITICAL FIX — Mutual Decoy Fallacy resolved:
    - The old code assigned pChEMBL=4.0 to UNTESTED subtypes for multi-target
      compounds, creating false negatives (missing-data != inactive).
    - Now, only explicit measured values from ChEMBL/GPCRdb and verified
      P2Y structural decoys are included.
    - Compounds missing data for a subtype are simply absent from that
      subtype's training set (NaN gap — handled later by feature filtering).
    """
    if save_lookup_path is None:
        save_lookup_path = str(PROCESSED_DATA_DIR / "db_lookup.json")

    logger.info("Loading data from %s (role=%s, endpoint=%s, decoys=%s)",
                 data_dir, target_role, target_endpoint, include_decoys)

    df = load_all_raw_data(data_dir)

    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()
    df["standard_relation"] = df["standard_relation"].astype(str).str.strip()
    df["assay_type"] = df["assay_type"].astype(str).str.upper().str.strip()
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")

    initial_count = len(df)

    df = df[
        (df["standard_relation"] == "=") &
        (df["confidence_score"] >= REQUIRED_CONFIDENCE) &
        (df["assay_type"].isin({"B", "F"})) &
        (df["standard_type"].isin(VALID_STANDARD_TYPES)) &
        (df["pchembl_value"].notna())
    ].copy()

    if target_endpoint != "all":
        if target_endpoint.upper() in {"KI", "KD"}:
            df = df[df["standard_type"].isin({"KI", "KD"})].copy()
        elif target_endpoint.upper() in {"IC50", "EC50"}:
            df = df[df["standard_type"].isin({"IC50", "EC50"})].copy()

    df["TAG"] = df["TAG"].astype(str).str.strip()
    df["target_subtype"] = df["TAG"].map(SUBTYPE_MAP)
    df = df[df["target_subtype"].notna()].copy()

    df["canonical_smiles"] = df["smiles"].apply(canonicalize)
    df = df[df["canonical_smiles"].notna()].copy()

    post_filter_count = len(df)
    logger.info("Scientific filters: %d raw rows -> %d high-quality rows.", initial_count, post_filter_count)

    # Vectorized known_targets building (replaces iterrows)
    known_targets = {}
    _hits_col = df["targets_hit"].fillna("").astype(str)
    _tag_col = df["TAG"].astype(str)
    _smi_col = df["canonical_smiles"]
    for smi, hits, tag in zip(_smi_col.values, _hits_col.values, _tag_col.values):
        subtypes = set()
        for hit in hits.split(";"):
            hit = hit.strip()
            if hit in SUBTYPE_MAP:
                subtypes.add(SUBTYPE_MAP[hit])
        if tag in SUBTYPE_MAP:
            subtypes.add(SUBTYPE_MAP[tag])
        if smi not in known_targets:
            known_targets[smi] = set()
        known_targets[smi].update(subtypes)

    barcodes = []
    for smi in df["canonical_smiles"]:
        barcode = canonicalize(smi)
        barcodes.append(barcode)
    df["barcode"] = barcodes

    df["_priority"] = np.where(df["standard_type"].isin({"KI", "KD"}), 0, 1)
    min_priority = df.groupby(["barcode", "target_subtype"])["_priority"].transform("min")
    best_df = df[df["_priority"] == min_priority]

    medians = best_df.groupby(["barcode", "target_subtype"])["pchembl_value"].median().reset_index()
    orig_maxes = df.groupby(["barcode", "target_subtype"])["pchembl_value"].max().reset_index()

    shifts_df = pd.merge(medians, orig_maxes, on=["barcode", "target_subtype"], suffixes=("_med", "_max"))
    shifts = (shifts_df["pchembl_value_max"] - shifts_df["pchembl_value_med"]).abs().tolist()

    df_deduped = best_df.drop_duplicates(subset=["barcode", "target_subtype"], keep="first").copy()
    df_deduped = df_deduped.drop(columns=["pchembl_value", "_priority"])
    df_deduped = pd.merge(df_deduped, medians, on=["barcode", "target_subtype"]).reset_index(drop=True)
    df = df.drop(columns=["_priority"])
    final_count = len(df_deduped)

    mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
    max_shift = max(shifts) if shifts else 0.0

    n_unique_barcodes = df_deduped["barcode"].nunique()
    logger.info("Barcode dedup: %d rows -> %d unique (barcode, subtype) pairs.", post_filter_count, final_count)
    logger.info("Unique molecular barcodes: %d", n_unique_barcodes)
    logger.info("Collapse stats: mean shift=%.3f, max shift=%.3f", mean_shift, max_shift)

    # Vectorized lookup building (replaces iterrows)
    lookup = (
        df_deduped.groupby("canonical_smiles")
        .apply(lambda g: dict(zip(g["target_subtype"], g["pchembl_value"].astype(float))),
               include_groups=False)
        .to_dict()
    )

    if include_decoys:
        logger.info("Loading P2Y structural decoys...")
        decoy_rows = []

        p2y_path = PROCESSED_DATA_DIR / "p2y_decoys.csv"
        if p2y_path.exists():
            logger.info("Ingesting P2Y decoys from %s", p2y_path)
            p2y_df = pd.read_csv(p2y_path)
            # Vectorized barcode registration
            barcodes = [canonicalize(s) for s in p2y_df["canonical_smiles"].values]
            # Vectorized decoy row construction via cross-join
            for idx, (smi, bc) in enumerate(zip(p2y_df["canonical_smiles"].values, barcodes)):
                for sub in SUBTYPES:
                    decoy_rows.append({
                        "TAG": f"{sub}R",
                        "canonical_smiles": smi,
                        "pchembl_value": DECOY_PCHEMBL,
                        "target_subtype": sub,
                        "standard_type": "DECOY_P2Y",
                        "barcode": bc,
                    })
            logger.info("Ingested %d P2Y structural decoy rows.", len(decoy_rows))

        if decoy_rows:
            decoy_df = pd.DataFrame(decoy_rows)
            df_deduped = pd.concat([df_deduped, decoy_df], ignore_index=True)
            logger.info("Total decoy rows after concat: %d", len(decoy_df))

            # Vectorized lookup rebuild after decoy concat
            lookup = (
                df_deduped.groupby("canonical_smiles")
                .apply(lambda g: dict(zip(g["target_subtype"], g["pchembl_value"].astype(float))),
                       include_groups=False)
                .to_dict()
            )



    Path(save_lookup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    keep_cols = ["TAG", "canonical_smiles", "pchembl_value", "target_subtype", "standard_type", "barcode"]
    available_cols = [c for c in keep_cols if c in df_deduped.columns]
    df_deduped = df_deduped[available_cols].copy()

    return df_deduped, lookup


if __name__ == "__main__":
    df, lookup = load_and_clean("data/raw", mode="precise", include_decoys=True)
    logger.info("Final dataset: %d rows, %d unique SMILES", len(df), df["canonical_smiles"].nunique())
    logger.info("Per-subtype distribution:\n%s", df["target_subtype"].value_counts().to_string())
