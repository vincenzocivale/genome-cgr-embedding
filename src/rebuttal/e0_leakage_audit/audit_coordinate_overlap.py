"""
Rebuttal E0 — Step 2: coordinate-overlap audit for the Genomics Long-Range Benchmark.

Reviewer e1HN asks whether splits contain overlapping genomic regions. The LRB
datasets carry genomic coordinates (`CHROM` + `START/STOP` or `POS`) and a predefined
`split` column, so we can measure region-level leakage directly:

  * shared chromosomes between train and test,
  * for interval datasets: % of test intervals overlapping any train interval,
  * for point datasets:    % of test positions within a window (default 1 kb) of a
    train position on the same chromosome, and exact position collisions.

A "donor / individual" note is recorded per dataset (the split axis is genomic region,
not per-donor, for every LRB task; see notes below).

Output: results/rebuttal/E0_data_leakage/coordinate_overlap_per_dataset.csv

Usage:
    python3 src/rebuttal/e0_leakage_audit/audit_coordinate_overlap.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

LRA_DIR = "data/genomics-long-range-benchmark"
OUT_CSV = "results/rebuttal/E0_data_leakage/coordinate_overlap_per_dataset.csv"

# dataset -> (relative csv, kind, donor/individual note, used_in_paper)
DATASETS = {
    "cage_prediction": (
        "cage_prediction/sequences_coordinates.csv", "interval",
        "Region-level split (genomic windows); no per-donor axis (CAGE targets aggregated).",
        True),
    "bulk_rna_expression": (
        "bulk_rna_expression/gene_coordinates.csv", "point_tss",
        "Per-gene split; GTEx expression aggregated across donors, no per-donor leakage axis.",
        True),
    "regulatory_elements_promoter": (
        "regulatory_elements/promoter_dataset.csv", "interval",
        "Region-level split; no donor axis.", False),
    "regulatory_elements_enhancer": (
        "regulatory_elements/enhancer_dataset.csv", "interval",
        "Region-level split; no donor axis.", False),
    "variant_effect_causal_eqtl": (
        "variant_effect_causal_eqtl/All_Tissues.csv", "point",
        "Variant-level split; GTEx eQTL across tissues, split axis is genomic position.",
        False),
    "variant_effect_pathogenic_coding": (
        "variant_effect_pathogenic/vep_pathogenic_coding.csv", "point",
        "Variant-level split (ClinVar/COSMIC); patient-derived but split by chromosome.",
        False),
    "chromatin_features": (
        "chromatin_features/histones_and_dnase.csv", "point",
        "Position-level split; no donor axis.", False),
}

WINDOW_BP = 1000  # proximity window for point datasets


def _cols(df):
    """Return case-insensitive column lookup {lower_name: actual_name}."""
    return {c.lower(): c for c in df.columns}


def _load(path, kind):
    """Load only the coordinate + split columns needed for `kind`."""
    head = pd.read_csv(path, nrows=1)
    lc = _cols(head)
    chrom = lc["chrom"]
    split = lc["split"]
    if kind == "interval":
        start, stop = lc["start"], lc["stop"]
        use = [chrom, split, start, stop]
        df = pd.read_csv(path, usecols=use)
        df = df.rename(columns={chrom: "chrom", split: "split", start: "start", stop: "stop"})
    elif kind == "point":
        pos = lc["pos"]
        use = [chrom, split, pos]
        df = pd.read_csv(path, usecols=use)
        df = df.rename(columns={chrom: "chrom", split: "split", pos: "pos"})
    elif kind == "point_tss":
        pos = lc["tss"]
        use = [chrom, split, pos]
        df = pd.read_csv(path, usecols=use)
        df = df.rename(columns={chrom: "chrom", split: "split", pos: "pos"})
    df["chrom"] = df["chrom"].astype(str)
    return df


def _interval_overlap_pct(train, test):
    """% of test intervals overlapping any train interval on the same chromosome,
    plus the gap (bp) from each test interval to the nearest train interval."""
    if len(test) == 0:
        return np.nan, 0, np.nan, np.nan
    hits = 0
    gaps = []
    for c, te in test.groupby("chrom"):
        tr = train[train["chrom"] == c]
        if len(tr) == 0:
            continue
        tr = tr.sort_values("start")
        tr_start = tr["start"].to_numpy()
        tr_end = tr["stop"].to_numpy()
        tr_cummax_end = np.maximum.accumulate(tr_end)
        te_s = te["start"].to_numpy()
        te_e = te["stop"].to_numpy()
        # overlap: any train interval with start <= test.end and (cummax end) >= test.start
        idx = np.searchsorted(tr_start, te_e, side="right") - 1
        valid = idx >= 0
        ov = np.zeros(len(te_s), dtype=bool)
        ov[valid] = tr_cummax_end[idx[valid]] >= te_s[valid]
        hits += int(ov.sum())
        # nearest gap for non-overlapping test intervals
        ts = np.sort(tr_start)
        te_srt = np.sort(tr_end)
        for s0, e0, is_ov in zip(te_s, te_e, ov):
            if is_ov:
                gaps.append(0)
                continue
            left = te_srt[te_srt <= s0]
            right = ts[ts >= e0]
            g = min((s0 - left.max()) if left.size else np.inf,
                    (right.min() - e0) if right.size else np.inf)
            gaps.append(g)
    gaps = np.array([g for g in gaps if np.isfinite(g)]) if gaps else np.array([])
    min_gap = float(gaps.min()) if gaps.size else np.nan
    frac_100kb = float((gaps < 100_000).mean()) if gaps.size else np.nan
    return 100.0 * hits / len(test), hits, min_gap, frac_100kb


def _point_overlap_pct(train, test, window=WINDOW_BP):
    """% of test positions within `window` bp of a same-chromosome train position,
    and count of exact position collisions."""
    if len(test) == 0:
        return np.nan, np.nan, 0, 0
    within = 0
    exact = 0
    for c, te in test.groupby("chrom"):
        tr = train[train["chrom"] == c]
        if len(tr) == 0:
            continue
        tp = np.sort(tr["pos"].to_numpy())
        qp = te["pos"].to_numpy()
        idx = np.searchsorted(tp, qp)
        # distance to nearest train position (left/right neighbour)
        left = np.where(idx > 0, qp - tp[np.clip(idx - 1, 0, len(tp) - 1)], np.inf)
        right = np.where(idx < len(tp), tp[np.clip(idx, 0, len(tp) - 1)] - qp, np.inf)
        dist = np.minimum(left, right)
        within += int((dist <= window).sum())
        exact += int((dist == 0).sum())
    n = len(test)
    return 100.0 * within / n, 100.0 * exact / n, within, exact


def main():
    ap = argparse.ArgumentParser(description="E0 LRB coordinate-overlap audit")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--window", type=int, default=WINDOW_BP)
    args = ap.parse_args()

    rows = []
    for name, (rel, kind, donor_note, in_paper) in DATASETS.items():
        path = os.path.join(LRA_DIR, rel)
        if not os.path.exists(path):
            print(f"  {name:40s} MISSING ({rel})")
            continue
        df = _load(path, kind)
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]
        tr_ch = set(train["chrom"].unique())
        te_ch = set(test["chrom"].unique())
        shared = sorted(tr_ch & te_ch)

        row = {
            "dataset": name, "csv": rel, "kind": kind, "used_in_paper": in_paper,
            "n_train": len(train), "n_test": len(test),
            "n_chrom_train": len(tr_ch), "n_chrom_test": len(te_ch),
            "shared_chrom_count": len(shared),
            "shared_chrom": ",".join(shared) if shared else "",
            "donor_note": donor_note,
        }

        if len(test) == 0 or len(train) == 0:
            row.update({"pct_test_interval_overlap_train": np.nan,
                        "pct_test_pos_within_window": np.nan,
                        "pct_test_pos_exact": np.nan,
                        "leakage_safe": (len(shared) == 0)})
            note = "no train or no test split -> cross-check N/A"
        elif len(shared) == 0:
            # Chromosome-disjoint by construction -> zero region overlap.
            row.update({"pct_test_interval_overlap_train": 0.0,
                        "pct_test_pos_within_window": 0.0,
                        "pct_test_pos_exact": 0.0,
                        "leakage_safe": True})
            note = "chromosome-disjoint"
        else:
            if kind == "interval":
                pct, hits, min_gap, frac_100kb = _interval_overlap_pct(train, test)
                row.update({"pct_test_interval_overlap_train": round(pct, 4),
                            "min_gap_to_train_bp": min_gap,
                            "frac_test_within_100kb_train": round(frac_100kb, 4)
                                if frac_100kb == frac_100kb else np.nan,
                            "pct_test_pos_within_window": np.nan,
                            "pct_test_pos_exact": np.nan,
                            "leakage_safe": (pct == 0)})
            else:
                pw, pe, nw, ne = _point_overlap_pct(train, test, args.window)
                row.update({"pct_test_interval_overlap_train": np.nan,
                            "pct_test_pos_within_window": round(pw, 4),
                            "pct_test_pos_exact": round(pe, 4),
                            "leakage_safe": (pe == 0)})
            note = f"{len(shared)} shared chrom -> measured overlap"

        rows.append(row)
        print(f"  {name:40s} shared_chrom={len(shared):2d}  "
              f"interval_ov={row.get('pct_test_interval_overlap_train')}  "
              f"pos_within{args.window}bp={row.get('pct_test_pos_within_window')}  "
              f"[{note}]")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
