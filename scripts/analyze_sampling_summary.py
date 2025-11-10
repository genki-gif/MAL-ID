#!/usr/bin/env python
import os
from pathlib import Path
from typing import Dict, Any

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


def main():
    sequences_path = config.paths.sequences
    sequences_sampled_path = config.paths.sequences_sampled

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


