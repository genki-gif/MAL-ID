#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import dask.dataframe as dd
from malid import config, logger


def resolve_paths(dataset_version: Optional[str]) -> Dict[str, Path]:
	# Resolve dataset version (CLI > ENV > config default)
	ds_ver = (
		dataset_version
		if dataset_version is not None
		else os.getenv("MALID_DATASET_VERSION", config.dataset_version)
	)
	paths = config.make_paths(
		embedder=config.embedder,
		cross_validation_split_strategy=config.cross_validation_split_strategy,
		dataset_version=ds_ver,
		base_data_dir=config.paths.base_data_dir,
		base_output_dir=config.paths.base_output_dir,
		base_scratch_dir=config.paths.base_scratch_dir,
	)
	return {
		"sequences": paths.sequences,
		"outdir": paths.base_output_dir / "sampling_summary" / "clone_counts",
	}


def _choose_clone_id_column(df: dd.DataFrame) -> Optional[str]:
	# Prefer canonical clone identifier if present
	candidates = ["clone_id", "clone", "clone_uid"]
	for c in candidates:
		if c in df.columns:
			return c
	return None


def _choose_isotype_column(df: dd.DataFrame) -> Optional[str]:
	candidates = ["extracted_isotype", "isotype", "isotype_inferred"]
	for c in candidates:
		if c in df.columns:
			return c
	return None


def compute_counts(
	df: dd.DataFrame,
	group_cols: List[str],
	clone_col: Optional[str],
) -> dd.DataFrame:
	# Dask does not support pandas-style named aggregations with ('col', 'size')
	# Compute size via groupby.size() and nunique via a separate aggregation, then merge.
	gb = df.groupby(group_cols)
	seq_count = gb.size().rename("__seq_count__").to_frame().reset_index()
	if clone_col is not None and clone_col in df.columns:
		clone_count = gb[clone_col].nunique().rename("__clone_count__").reset_index()
		out = seq_count.merge(clone_count, on=group_cols, how="left")
	else:
		out = seq_count
	return out


def to_pandas(df: dd.DataFrame):
	# Persist then compute to pandas
	return df.compute()


def save_tsv(df, path: Path):
	import pandas as pd  # type: ignore
	path.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(df).to_csv(path, sep="\t", index=False)


def save_percentiles(series, quantiles: List[float], path: Path, name: str):
	import pandas as pd  # type: ignore
	q = pd.Series(series).quantile(quantiles)
	out = q.reset_index()
	out.columns = ["quantile", name]
	path.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(path, sep="\t", index=False)


def plot_hist(series, title: str, xlabel: str, path: Path):
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		import numpy as np
	except Exception as err:
		logger.warning(f"Plotting skipped: {err}")
		return

	path.parent.mkdir(parents=True, exist_ok=True)
	data = series.dropna()
	# Use log-binning for wide distributions
	data = data[data >= 0]
	data = np.asarray(data, dtype=float)
	if data.size == 0:
		logger.warning(f"No data to plot for {title}")
		return
	plt.figure(figsize=(7, 4))
	plt.hist(data, bins=100)
	plt.yscale("linear")
	plt.xscale("log")
	plt.title(title)
	plt.xlabel(xlabel + " (log scale)")
	plt.ylabel("specimen count")
	plt.tight_layout()
	plt.savefig(path, dpi=120)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Visualize clone/sequence count distributions per specimen.")
	parser.add_argument("--dataset-version", type=str, default=None, help="e.g. 20231109")
	parser.add_argument("--by-disease", action="store_true", help="Emit per-disease TSVs as well")
	args = parser.parse_args()

	paths = resolve_paths(args.dataset_version)
	sequences_path: Path = paths["sequences"]
	outdir: Path = paths["outdir"]

	logger.info(f"Reading full sequences parquet: {sequences_path}")
	df = dd.read_parquet(sequences_path, engine="pyarrow")

	clone_col = _choose_clone_id_column(df)
	isotype_col = _choose_isotype_column(df)
	has_disease = "disease" in df.columns

	# Overall per-specimen counts
	group_cols = ["participant_label", "specimen_label"]
	if has_disease:
		group_cols += ["disease"]
	g_overall = compute_counts(df, group_cols, clone_col)
	pdf_overall = to_pandas(g_overall)
	save_tsv(pdf_overall, outdir / "counts_per_specimen.tsv")

	# Percentiles
	q_list = [0.5, 0.75, 0.9, 0.95, 0.99]
	if "__clone_count__" in pdf_overall.columns:
		save_percentiles(
			pdf_overall["__clone_count__"], q_list, outdir / "percentiles.clone_count.tsv", "clone_count"
		)
	save_percentiles(
		pdf_overall["__seq_count__"], q_list, outdir / "percentiles.sequence_count.tsv", "sequence_count"
	)

	# Histograms
	if "__clone_count__" in pdf_overall.columns:
		plot_hist(
			pdf_overall["__clone_count__"],
			"Unique clones per specimen",
			"unique clone count",
			outdir / "hist.clone_count.per_specimen.png",
		)
	plot_hist(
		pdf_overall["__seq_count__"],
		"Sequences per specimen",
		"sequence count",
		outdir / "hist.sequence_count.per_specimen.png",
	)

	# By-isotype per-specimen counts (if available)
	if isotype_col is not None:
		group_cols_iso = ["participant_label", "specimen_label", isotype_col]
		if has_disease:
			group_cols_iso += ["disease"]
		g_iso = compute_counts(df, group_cols_iso, clone_col)
		pdf_iso = to_pandas(g_iso)
		save_tsv(pdf_iso, outdir / "counts_per_specimen_by_isotype.tsv")

	# Optional: by-disease summaries
	if args.by_disease and has_disease:
		import pandas as pd  # type: ignore
		p = (
			pdf_overall.groupby("disease")[["__seq_count__"] + (["__clone_count__"] if "__clone_count__" in pdf_overall.columns else [])]
			.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
		)
		p.to_csv(outdir / "by_disease.summary.tsv", sep="\t")

	print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
	main()


