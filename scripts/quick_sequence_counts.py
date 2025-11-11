#!/usr/bin/env python
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pyarrow.dataset as pa_ds  # type: ignore
import pandas as pd  # type: ignore

from malid import config, logger


def resolve_sequences_path(dataset_version: Optional[str]) -> Path:
	# dataset_version: ENV > config default if arg not provided
	if dataset_version is None:
		dataset_version = os.getenv("MALID_DATASET_VERSION", config.dataset_version)
	paths = config.make_paths(
		embedder=config.embedder,
		cross_validation_split_strategy=config.cross_validation_split_strategy,
		dataset_version=dataset_version,
		base_data_dir=config.paths.base_data_dir,
		base_output_dir=config.paths.base_output_dir,
		base_scratch_dir=config.paths.base_scratch_dir,
	)
	return paths.sequences


def extract_partition_values_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
	# Expect .../participant_label=XXX/specimen_label=YYY/part-*.parquet
	# Robust fallback via regex
	p_label = None
	s_label = None
	m1 = re.search(r"/participant_label=([^/]+)/", path)
	if m1:
		p_label = m1.group(1)
	m2 = re.search(r"/specimen_label=([^/]+)/", path)
	if m2:
		s_label = m2.group(1)
	return p_label, s_label


def fast_count_rows_per_specimen(sequences_parquet_path: Path) -> pd.DataFrame:
	ds = pa_ds.dataset(str(sequences_parquet_path), format="parquet", partitioning="hive")
	records: Dict[Tuple[str, str], int] = {}
	for frag in ds.get_fragments():
		# Fast row counting without reading data
		try:
			num_rows = frag.count_rows()
		except Exception:
			# Fallback via file metadata
			scan = frag.scanner(projected_schema=None)
			num_rows = 0
			for task in scan.scan_batches():
				num_rows += task.num_rows
		path = getattr(frag, "path", "")
		p_label, s_label = extract_partition_values_from_path(path)
		if p_label is None or s_label is None:
			# As a fallback, try evaluating the partition expression string
			try:
				expr = str(frag.partition_expression)
				mp = re.search(r"participant_label == '([^']+)'", expr)
				ms = re.search(r"specimen_label == '([^']+)'", expr)
				if mp:
					p_label = mp.group(1)
				if ms:
					s_label = ms.group(1)
			except Exception:
				pass
		if p_label is None or s_label is None:
			# Skip if cannot resolve partition values
			continue
		key = (p_label, s_label)
		records[key] = records.get(key, 0) + int(num_rows)
	df = pd.DataFrame(
		[(p, s, c) for (p, s), c in records.items()],
		columns=["participant_label", "specimen_label", "__seq_count__"],
	)
	return df


def write_outputs(df: pd.DataFrame, outdir: Path):
	outdir.mkdir(parents=True, exist_ok=True)
	(df.sort_values("__seq_count__", ascending=False)).to_csv(
		outdir / "quick_counts_per_specimen.tsv", sep="\t", index=False
	)
	# Percentiles
	qs = [0.5, 0.75, 0.9, 0.95, 0.99]
	q = df["__seq_count__"].quantile(qs).reset_index()
	q.columns = ["quantile", "sequence_count"]
	q.to_csv(outdir / "quick_percentiles.sequence_count.tsv", sep="\t", index=False)


def main():
	sequences_path = resolve_sequences_path(dataset_version=None)
	outdir = config.paths.base_output_dir / "sampling_summary" / "quick_sequence_counts"
	logger.info(f"Counting rows per specimen from: {sequences_path}")
	df = fast_count_rows_per_specimen(sequences_path)
	logger.info(f"Resolved specimens: {df.shape[0]}")
	write_outputs(df, outdir)
	print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
	main()


