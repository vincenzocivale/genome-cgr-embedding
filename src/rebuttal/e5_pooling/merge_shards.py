"""Merge per-pooling shard CSVs produced by concurrent run_pooling_comparison.py
invocations back into one file, deduplicating by (dataset, model, pooling, probe).

Usage: python merge_shards.py <main.csv> <shard1.csv> [<shard2.csv> ...]

Note: do NOT invoke this via `conda run ... python - <<EOF` (heredoc into a
`-` stdin script) -- `conda run` does not reliably forward heredoc stdin, so
the script silently does nothing and no error is raised. Always call it as a
real file, as done here.
"""
import sys
import os
import pandas as pd

main_path = sys.argv[1]
shard_paths = sys.argv[2:]
frames = [pd.read_csv(p) for p in [main_path, *shard_paths] if os.path.exists(p)]
if not frames:
    print(f"ERROR: no input files found among {[main_path, *shard_paths]}", file=sys.stderr)
    sys.exit(1)
merged = pd.concat(frames, ignore_index=True)
key = ["dataset", "model", "pooling", "probe"]
before = len(merged)
merged = merged.drop_duplicates(subset=key, keep="last")
merged.to_csv(main_path, index=False)
print(f"merged {len(frames)} files ({before} rows before dedup) -> {len(merged)} rows written to {main_path}")
