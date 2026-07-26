"""Write N size-balanced dataset-name groups to <outdir>/group_{i}.txt.

Used by run_e5_dataset_sharded.sh to split the 57 benchmark datasets across
concurrent processes, greedily assigning each (largest-first) to the
currently least-loaded shard so big datasets are spread out.

Usage: python make_dataset_shards.py <data_root> <n_shards> <outdir>

Must be a real file, not a `conda run python - <<EOF` heredoc: conda run
does not forward heredoc stdin, so the script would silently do nothing.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from src.data.loader import discover_datasets

data_root, n, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]

sizes = []
for d in discover_datasets(data_root):
    try:
        with open(d["train_path"]) as f:
            ntr = sum(1 for _ in f) - 1
        with open(d["test_path"]) as f:
            nte = sum(1 for _ in f) - 1
    except OSError:
        ntr = nte = 0
    sizes.append((d["name"], ntr + nte))

sizes.sort(key=lambda x: -x[1])
groups = [[] for _ in range(n)]
loads = [0] * n
for name, sz in sizes:
    i = min(range(n), key=lambda k: loads[k])
    groups[i].append(name)
    loads[i] += sz

os.makedirs(outdir, exist_ok=True)
for i, g in enumerate(groups):
    with open(os.path.join(outdir, f"group_{i}.txt"), "w") as f:
        f.write(" ".join(g))
print(f"built {n} groups in {outdir}, loads={loads}")
