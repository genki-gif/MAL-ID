#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import dask.dataframe as dd

from malid import config, logger


def _safe_nunique(df: dd.DataFrame, col: str) -> int:
    if col in df.columns:
        return int(df[col].nunique().compute())
    return 0


def _safe_groupby_nunique_specimens_by_disease(df: dd.DataFrame) -> Dict[str, int]:
    # Return disease -> nunique(specimen_label), if columns exist; else empty dict
    required = {"disease", "specimen_label"}
    if not required.issubset(set(df.columns)):
        return {}
    grp = df.groupby("disease")["specimen_label"].nunique().compute()
    # Convert to plain dict with str keys
    return {str(k): int(v) for k, v in grp.items()}


def summarize_dataframe(df: dd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    # Row count
    summary["num_rows"] = int(df.shape[0].compute())
    # Entities
    summary["num_participants"] = _safe_nunique(df, "participant_label")
    summary["num_specimens"] = _safe_nunique(df, "specimen_label")
    # Optional: disease breakdown (by specimens)
    summary["by_disease_num_specimens"] = _safe_groupby_nunique_specimens_by_disease(df)
    return summary


def write_tsv(path: Path, rows: Dict[str, Any]) -> None:
    # Flatten dict-of-dicts (for disease breakdown) to two-column long TSVs next to summary TSV
    import pandas as pd

    # Top-level summary (excluding disease breakdowns)
    top = {
        k: v
        for k, v in rows.items()
        if k not in ["by_disease_num_specimens"]
    }
    df_top = pd.DataFrame([top])
    df_top.to_csv(path, sep="\t", index=False)

    # Disease breakdown
    disease_map = rows.get("by_disease_num_specimens", {})
    if isinstance(disease_map, dict) and len(disease_map) > 0:
        df_dis = (
            pd.Series(disease_map, name="num_specimens")
            .rename_axis("disease")
            .reset_index()
            .sort_values("num_specimens", ascending=False)
        )
        path2 = path.with_name(path.stem + ".by_disease.tsv")
        df_dis.to_csv(path2, sep="\t", index=False)


def _discover_existing_dataset_version(base_data_dir: Path) -> Optional[str]:
    """
    Look for data/data_v_*/sequences.parquet and return a version string if found.
    Prefer the most recently modified directory.
    """
    candidates = []
    try:
        for child in base_data_dir.glob("data_v_*"):
            if (child / "sequences.parquet").exists():
                candidates.append(child)
    except Exception:
        return None
    if not candidates:
        return None
    # Sort by mtime, descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # child.name like data_v_20231109 -> extract suffix
    name = candidates[0].name
    if name.startswith("data_v_"):
        return name.replace("data_v_", "")
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarize sampling before/after ETL sampling step.")
    parser.add_argument(
        "--dataset-version",
        dest="dataset_version",
        type=str,
        default=None,
        help="Override dataset version (e.g., 20231109). Falls back to MALID_DATASET_VERSION or config default.",
    )
    args = parser.parse_args()

    # Resolve dataset version to use:
    dataset_version = (
        args.dataset_version
        if args.dataset_version is not None
        else os.getenv("MALID_DATASET_VERSION", config.dataset_version)
    )

    # Build paths for the resolved dataset version
    paths_for_run = config.make_paths(
        embedder=config.embedder,
        cross_validation_split_strategy=config.cross_validation_split_strategy,
        dataset_version=dataset_version,
        base_data_dir=config.paths.base_data_dir,     # keep same roots
        base_output_dir=config.paths.base_output_dir,
        base_scratch_dir=config.paths.base_scratch_dir,
    )

    # If sequences are missing, try to auto-discover an existing dataset version
    if not (paths_for_run.sequences.exists()):
        discovered = _discover_existing_dataset_version(config.paths.base_data_dir)
        if discovered and discovered != dataset_version:
            logger.warning(
                f"Configured dataset version {dataset_version} not found, falling back to discovered existing version {discovered}"
            )
            dataset_version = discovered
            paths_for_run = config.make_paths(
                embedder=config.embedder,
                cross_validation_split_strategy=config.cross_validation_split_strategy,
                dataset_version=dataset_version,
                base_data_dir=config.paths.base_data_dir,
                base_output_dir=config.paths.base_output_dir,
                base_scratch_dir=config.paths.base_scratch_dir,
            )
        else:
            # Keep going; dd.read_parquet will raise a clear FileNotFoundError
            pass

    sequences_path = paths_for_run.sequences
    sequences_sampled_path = paths_for_run.sequences_sampled

    out_dir = config.paths.base_output_dir / "sampling_summary"
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Reading (all): {sequences_path}")
    df_all = dd.read_parquet(sequences_path, engine="pyarrow")
    logger.info(f"Reading (sampled): {sequences_sampled_path}")
    df_sampled = dd.read_parquet(sequences_sampled_path, engine="pyarrow")

    logger.info("Summarizing (all)")
    summary_all = summarize_dataframe(df_all)
    logger.info("Summarizing (sampled)")
    summary_sampled = summarize_dataframe(df_sampled)

    # Write outputs
    write_tsv(out_dir / "summary.all.tsv", summary_all)
    write_tsv(out_dir / "summary.sampled.tsv", summary_sampled)

    # Console print (succinct)
    def _fmt(s: Dict[str, Any]) -> str:
        return (
            f"rows={s.get('num_rows')}, "
            f"participants={s.get('num_participants')}, "
            f"specimens={s.get('num_specimens')}"
        )

    print("All:     " + _fmt(summary_all))
    print("Sampled: " + _fmt(summary_sampled))
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()


