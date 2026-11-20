"""Multi-source equilibrium binding (Ki/Kd -> pKi) loader, standardizer, and deduplicator."""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.chem_utils import canonicalize
from src.config import (
    CHEMBL_TARGETS,
    PROCESSED_DATA_DIR,
    REQUIRED_CONFIDENCE,
    SUBTYPES,
)

logger = logging.getLogger(__name__)

SUBTYPE_MAP = {
    "A1R": "A1",
    "A2AR": "A2A",
    "A2BR": "A2B",
    "A3R": "A3",
    "A1": "A1",
    "A2A": "A2A",
    "A2B": "A2B",
    "A3": "A3",
}

_CHEMBL_ID_TO_SUBTYPE = {v: k for k, v in CHEMBL_TARGETS.items()}
_GPCRDB_FILES = {st: f"GPCRdb_{st}.xlsx" for st in SUBTYPES}
_GPCRDB_VALID_TYPES = {"KI", "PKI", "KD", "PKD"}
_CHEMBL_VALID_TYPES = {"KI", "KD"}
_GPCRDB_FUNCTIONAL_TYPES = {"IC50", "PIC50", "EC50", "PEC50"}
_CHEMBL_FUNCTIONAL_TYPES = {"IC50", "EC50"}
_CONFLICT_SPREAD_THRESHOLD = 2.0
_canonicalize_smiles = canonicalize


def _normalize_binding_type(val: str) -> str:
    s = str(val).strip().upper()
    if s in {"KI", "PKI"}:
        return "KI"
    if s in {"KD", "PKD"}:
        return "KD"
    if s in {"IC50", "PIC50"}:
        return "IC50"
    if s in {"EC50", "PEC50"}:
        return "EC50"
    return s


def _load_gpcrdb_sources(raw_path: Path, valid_types: set[str] = _GPCRDB_VALID_TYPES) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for st, fname in _GPCRDB_FILES.items():
        fpath = raw_path / fname
        if not fpath.exists() or fpath.stat().st_size < 1000:
            continue
        df = pd.read_excel(fpath)
        species = df["Species"].astype(str).str.strip()
        relation = df["Activity Relation"].astype(str).str.strip()
        act_type = df["Activity Type"].astype(str).str.strip().str.upper()
        pval = pd.to_numeric(df["p-value (-log)"], errors="coerce")

        mask = (
            (species == "Human")
            & (relation == "=")
            & (act_type.isin(valid_types))
            & (pval.notna())
        )
        filtered = df.loc[mask].copy()
        if filtered.empty:
            continue

        out = pd.DataFrame(
            {
                "smiles": filtered["Smiles"].astype(str).values,
                "pchembl_value": pd.to_numeric(filtered["p-value (-log)"], errors="coerce").values,
                "target_subtype": st,
                "standard_type": [_normalize_binding_type(t) for t in filtered["Activity Type"].values],
                "standard_relation": "=",
                "source_db": "GPCRdb",
            }
        )
        logger.info("Loaded GPCRdb %s (%s): %d rows.", st, fname, len(out))
        frames.append(out)
    return frames


def _load_chembl_parents_source(raw_path: Path, valid_types: set[str] = _CHEMBL_VALID_TYPES) -> Optional[pd.DataFrame]:
    fpath = raw_path / "AR_all_unique_parents_with_smiles.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)

    relation = df["standard_relation"].astype(str).str.strip()
    std_type = df["standard_type"].astype(str).str.strip().str.upper()
    pval = pd.to_numeric(df["pchembl_value"], errors="coerce")

    mask = (
        (relation == "=")
        & (std_type.isin(valid_types))
        & (pval.notna())
    )
    if "confidence_score" in df.columns:
        conf = pd.to_numeric(df["confidence_score"], errors="coerce")
        mask = mask & (conf >= REQUIRED_CONFIDENCE)

    filtered = df.loc[mask].copy()
    if filtered.empty:
        return None

    if "target_subtype" in filtered.columns:
        subtypes = filtered["target_subtype"].astype(str).str.strip().map(SUBTYPE_MAP)
    elif "TAG" in filtered.columns:
        subtypes = filtered["TAG"].astype(str).str.strip().map(SUBTYPE_MAP)
    elif "target_chembl_id" in filtered.columns:
        subtypes = filtered["target_chembl_id"].astype(str).str.strip().map(_CHEMBL_ID_TO_SUBTYPE)
    else:
        raise ValueError("AR_all_unique_parents_with_smiles.csv missing subtype identifier column.")

    out = pd.DataFrame(
        {
            "smiles": filtered["smiles"].astype(str).values,
            "pchembl_value": pd.to_numeric(filtered["pchembl_value"], errors="coerce").values,
            "target_subtype": subtypes.values,
            "standard_type": filtered["standard_type"].astype(str).str.strip().str.upper().values,
            "standard_relation": "=",
            "source_db": "ChEMBL_Parents",
        }
    )
    logger.info("Loaded ChEMBL Parents (%s): %d rows.", fpath.name, len(out))
    return out


def _load_chembl_api_source(raw_path: Path, valid_types: set[str] = _CHEMBL_VALID_TYPES) -> Optional[pd.DataFrame]:
    fpath = raw_path / "chembl_human_ar_bioactivities.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)

    relation = df["standard_relation"].astype(str).str.strip()
    std_type = df["standard_type"].astype(str).str.strip().str.upper()
    pval = pd.to_numeric(df["pchembl_value"], errors="coerce")

    mask = (
        (relation == "=")
        & (std_type.isin(valid_types))
        & (pval.notna())
    )
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return None

    if "target_subtype" in filtered.columns:
        subtypes = filtered["target_subtype"].astype(str).str.strip().map(SUBTYPE_MAP)
    elif "target_chembl_id" in filtered.columns:
        subtypes = filtered["target_chembl_id"].astype(str).str.strip().map(_CHEMBL_ID_TO_SUBTYPE)
    elif "TAG" in filtered.columns:
        subtypes = filtered["TAG"].astype(str).str.strip().map(SUBTYPE_MAP)
    else:
        raise ValueError("chembl_human_ar_bioactivities.csv missing subtype identifier column.")

    out = pd.DataFrame(
        {
            "smiles": filtered["smiles"].astype(str).values,
            "pchembl_value": pd.to_numeric(filtered["pchembl_value"], errors="coerce").values,
            "target_subtype": subtypes.values,
            "standard_type": filtered["standard_type"].astype(str).str.strip().str.upper().values,
            "standard_relation": "=",
            "source_db": "ChEMBL_API",
        }
    )
    logger.info("Loaded ChEMBL API (%s): %d equilibrium Ki/Kd rows.", fpath.name, len(out))
    return out


def load_all_raw_data(data_dir: str | Path, assay_mode: str = "binding") -> pd.DataFrame:
    """Load and merge all verified raw sources (GPCRdb, ChEMBL_Parents, ChEMBL_API) for binding or functional mode."""
    raw_path = Path(data_dir)
    frames: list[pd.DataFrame] = []
    is_func = str(assay_mode).lower().strip() == "functional"
    gpcrdb_types = _GPCRDB_FUNCTIONAL_TYPES if is_func else _GPCRDB_VALID_TYPES
    chembl_types = _CHEMBL_FUNCTIONAL_TYPES if is_func else _CHEMBL_VALID_TYPES

    frames.extend(_load_gpcrdb_sources(raw_path, valid_types=gpcrdb_types))

    parents_df = _load_chembl_parents_source(raw_path, valid_types=chembl_types)
    if parents_df is not None:
        frames.append(parents_df)

    api_df = _load_chembl_api_source(raw_path, valid_types=chembl_types)
    if api_df is not None:
        frames.append(api_df)

    if not frames:
        raise ValueError(f"No valid datasets found in {data_dir} for assay_mode={assay_mode}")

    merged = pd.concat(frames, ignore_index=True)
    logger.info("Merged %d raw rows across %d source tables (assay_mode=%s).", len(merged), len(frames), assay_mode)
    return merged


def _build_lookup(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for smi, sub, val in zip(
        df["canonical_smiles"].values,
        df["target_subtype"].values,
        df["pchembl_value"].values,
    ):
        lookup.setdefault(smi, {})[sub] = float(val)
    return lookup


def load_and_clean(
    data_dir: str | Path = "data/raw",
    save_lookup_path: str | Path | None = None,
    mode: str = "precise",
    target_role: str = "all",
    target_endpoint: str = "all",
    include_decoys: bool = False,
    save_subtype_csvs: bool = False,
    subtype_csv_dir: str | Path | None = None,
    assay_mode: str = "binding",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Load, filter (binding Ki/Kd or functional IC50/EC50), canonicalize, conflict-filter, and median-aggregate AR data."""
    if include_decoys:
        logger.warning("Synthetic decoys are disabled by scientific policy; ignoring include_decoys=True.")

    is_func = str(assay_mode).lower().strip() == "functional"
    valid_chembl_types = _CHEMBL_FUNCTIONAL_TYPES if is_func else _CHEMBL_VALID_TYPES

    logger.info(
        "Loading %s data from %s (mode=%s, role=%s, endpoint=%s)",
        "functional IC50/EC50" if is_func else "equilibrium Ki/Kd",
        data_dir,
        mode,
        target_role,
        target_endpoint,
    )

    df = load_all_raw_data(data_dir, assay_mode=assay_mode)
    raw_count = len(df)
    raw_sources_breakdown = {
        str(k): int(v) for k, v in df["source_db"].value_counts().items()
    }

    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()
    df["standard_relation"] = df["standard_relation"].astype(str).str.strip()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["target_subtype"] = df["target_subtype"].astype(str).str.strip().map(SUBTYPE_MAP)

    mask = (
        (df["standard_relation"] == "=")
        & (df["standard_type"].isin(valid_chembl_types))
        & (df["target_subtype"].isin(SUBTYPES))
        & (df["pchembl_value"].notna())
        & (df["pchembl_value"] >= 3.0)
        & (df["pchembl_value"] <= 12.5)
    )
    df = df.loc[mask].copy()

    if target_endpoint != "all":
        ep = target_endpoint.upper().strip()
        if ep in valid_chembl_types:
            df = df[df["standard_type"] == ep].copy()

    unique_raw_smiles = df["smiles"].dropna().unique()
    canon_map = {s: canonicalize(s) for s in unique_raw_smiles}
    df["canonical_smiles"] = df["smiles"].map(canon_map)
    df = df[df["canonical_smiles"].notna()].copy()

    post_filter_count = len(df)
    logger.info(
        "Quality & canonicalization filters: %d merged rows -> %d valid canonical rows.",
        raw_count,
        post_filter_count,
    )

    primary_ep = "IC50" if is_func else "KI"
    secondary_ep = "EC50" if is_func else "KD"
    grouped = df.groupby(["canonical_smiles", "target_subtype"], sort=True)
    agg = grouped.agg(
        pKi_median=("pchembl_value", "median"),
        pKi_min=("pchembl_value", "min"),
        pKi_max=("pchembl_value", "max"),
        pKi_q25=("pchembl_value", lambda s: float(s.quantile(0.25))),
        pKi_q75=("pchembl_value", lambda s: float(s.quantile(0.75))),
        n_replicates=("pchembl_value", "size"),
        sources=("source_db", lambda s: ";".join(sorted(set(s.astype(str))))),
        standard_type=("standard_type", lambda s: primary_ep if primary_ep in set(s) else secondary_ep),
    ).reset_index()

    agg["spread"] = agg["pKi_max"] - agg["pKi_min"]
    agg["iqr"] = agg["pKi_q75"] - agg["pKi_q25"]
    conflict_mask = (
        ((agg["n_replicates"] == 2) & (agg["spread"] > _CONFLICT_SPREAD_THRESHOLD))
        | ((agg["n_replicates"] >= 3) & (agg["iqr"] > 1.5))
    )
    n_dropped_conflict = int(conflict_mask.sum())
    if n_dropped_conflict > 0:
        logger.info(
            "Dropped %d severely conflicting (molecule, subtype) groups with replicate spread > %.1f log units.",
            n_dropped_conflict,
            _CONFLICT_SPREAD_THRESHOLD,
        )

    df_deduped = agg.loc[~conflict_mask].copy().reset_index(drop=True)
    df_deduped["pKi"] = df_deduped["pKi_median"].astype(float)
    df_deduped["pchembl_value"] = df_deduped["pKi"]
    df_deduped["TAG"] = df_deduped["target_subtype"].astype(str) + "R"
    df_deduped["barcode"] = df_deduped["canonical_smiles"]

    logger.info(
        "Deduplication: %d rows -> %d unique (molecule, subtype) pairs across %d compounds (dropped %d conflicts).",
        post_filter_count,
        len(df_deduped),
        df_deduped["canonical_smiles"].nunique(),
        n_dropped_conflict,
    )

    if save_subtype_csvs or subtype_csv_dir is not None:
        out_dir = Path(subtype_csv_dir) if subtype_csv_dir is not None else PROCESSED_DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "functional" if is_func else "pki"
        csv_cols = ["canonical_smiles", "pKi", "target_subtype", "n_replicates", "sources"]
        for st in SUBTYPES:
            sub_df = df_deduped.loc[df_deduped["target_subtype"] == st, csv_cols].reset_index(drop=True)
            sub_path = out_dir / f"{st}_{suffix}.csv"
            sub_df.to_csv(sub_path, index=False)
            logger.info("Saved %s subtype dataset (%d rows) to %s", st, len(sub_df), sub_path)

    lookup = _build_lookup(df_deduped)

    if save_lookup_path is not None:
        out_path = Path(save_lookup_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f, indent=2, sort_keys=True)
        logger.info("Saved DB lookup (%d molecules) to %s", len(lookup), out_path)

    keep_cols = [
        "TAG",
        "canonical_smiles",
        "pchembl_value",
        "pKi",
        "target_subtype",
        "standard_type",
        "n_replicates",
        "sources",
        "barcode",
    ]
    out_df = df_deduped[keep_cols].copy()
    out_df.attrs["assay_filter"] = (
        "Functional IC50 and EC50 only (pIC50/pEC50)"
        if is_func
        else "Equilibrium binding Ki and Kd only (pKi)"
    )
    out_df.attrs["n_raw_records"] = int(raw_count)
    out_df.attrs["n_replicates_dropped_conflict"] = n_dropped_conflict
    out_df.attrs["raw_sources_breakdown"] = raw_sources_breakdown
    return out_df, lookup


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cleaned_df, _ = load_and_clean("data/raw", save_lookup_path=None, include_decoys=False)
    logger.info(
        "Final dataset: %d rows, %d unique molecules (conflicts dropped: %d)",
        len(cleaned_df),
        cleaned_df["canonical_smiles"].nunique(),
        cleaned_df.attrs.get("n_replicates_dropped_conflict", 0),
    )
    logger.info("Per-subtype counts:\n%s", cleaned_df["target_subtype"].value_counts().to_string())
