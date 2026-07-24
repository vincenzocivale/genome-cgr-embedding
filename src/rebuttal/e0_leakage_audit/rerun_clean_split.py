"""
Rebuttal E0 — Step 5: re-run the main results on the leakage-resistant (clean) test.

For each high-risk classification dataset we keep the training set fixed, train the
main models once, and evaluate each on (a) the original published test set and (b) the
de-contaminated test set from Step 4. Because the model is identical, the difference in
score isolates the contribution of leaked (duplicated) test rows.

Models re-run:
  * k-mer Random Forest        (k = 4, 5, 6; FCGR features)
  * FM Random Forest           (cached embeddings, mean pooling)
  * FM linear probe            (cached embeddings, mean pooling)
FM embeddings are re-indexed from the on-disk cache (CSV row order) — no GPU re-embedding.

Output: results/rebuttal/E0_data_leakage/rerun_original_vs_clean.csv

Usage:
    python3 src/rebuttal/e0_leakage_audit/rerun_clean_split.py --n-jobs 8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import load_dataset
from src.core.fcgr import batch_fcgr
from src.features.kmer_features import kmer_from_grids
from src.training.rf_pipeline import train_rf_fast, eval_rf
from src.training.linear_pipeline import train_linear_classifier, eval_linear_classifier

RESULT_DIR = "results/rebuttal/E0_data_leakage"
CLEAN_DIR = os.path.join(RESULT_DIR, "clean_splits")
HIGH_TXT = os.path.join(RESULT_DIR, "high_risk_datasets.txt")
OUT_CSV = os.path.join(RESULT_DIR, "rerun_original_vs_clean.csv")
DATA_ROOT = "data/dna_foundation_benchmark"

# Cached FM embeddings (mean pooling = paper default). Candidate dirs tried in order;
# {ds} is the dataset name with "/" replaced by "__". First dir with train+test wins.
FM_MODELS = {
    "NTv3_650M_pre": [
        "cache/fm_embeddings/InstaDeepAI__NTv3_650M_pre/{ds}",
        "cache/fm_embeddings/InstaDeepAI__NTv3_650M_pre/pooling_mean/{ds}",
    ],
    "hyenadna-medium-160k": [
        "cache/hyena_embeddings/{ds}",
        "cache/hyena_embeddings/pooling_mean/{ds}",
    ],
    "DNABERT-2-117M": [
        "cache/fm_embeddings/zhihan1996__DNABERT-2-117M/pooling_mean/{ds}",
        "cache/fm_embeddings/zhihan1996__DNABERT-2-117M/{ds}",
    ],
    "evo2_1b_base": [   # bonus: 4th model added during the current work, if cached
        "cache/evo2_1b_base/{ds}",
    ],
}


def _safe_ds(name):
    return name.replace("/", "__").replace("\\", "__")


def _find_fm(model, name):
    ds = _safe_ds(name)
    for tmpl in FM_MODELS[model]:
        d = tmpl.format(ds=ds)
        tr, te = os.path.join(d, "train.npz"), os.path.join(d, "test.npz")
        if os.path.exists(tr) and os.path.exists(te):
            return tr, te
    return None, None


def _load_npz(path):
    z = np.load(path)
    return z["embeddings"].astype(np.float32)


# Tier 2 (near-dup-removed) test is only scored if it retains at least this fraction
# and this many rows — otherwise the subset is too small to be meaningful.
MIN_TIER2_FRAC = 0.10
MIN_TIER2_ROWS = 100


def _eval(model, X, y, n_classes, kind):
    """Class-safe evaluation: if a test subset does not contain all `n_classes`
    classes (which breaks multiclass AUROC), retry with AUROC set to NaN."""
    try:
        return (eval_rf(model, X, y, n_classes) if kind == "rf"
                else eval_linear_classifier(model, X, y, n_classes))
    except ValueError:
        from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
        y_pred = model.predict(X)
        return {"MCC": matthews_corrcoef(y, y_pred),
                "AUROC": np.nan,
                "F1": f1_score(y, y_pred, average="macro"),
                "Accuracy": accuracy_score(y, y_pred)}


def _rows_from(dataset, feature, kind, model, X_test, y_test,
               keep_exact, keep_neardup, n_test, n_classes):
    """Score `model` on original / Tier-1 / Tier-2 test sets and return long rows.
    `n_classes` is the number of classes the model was trained on."""
    orig = _eval(model, X_test, y_test, n_classes, kind)
    # Tier 1
    ke = keep_exact
    t1 = _eval(model, X_test[ke], y_test[ke], n_classes, kind)
    n_t1 = len(ke)
    # Tier 2 (guarded: enough rows, enough fraction, all classes present)
    kn = keep_neardup
    do_t2 = (len(kn) >= MIN_TIER2_ROWS) and (len(kn) >= MIN_TIER2_FRAC * n_test) \
        and (len(set(y_test[kn])) == n_classes)
    t2 = _eval(model, X_test[kn], y_test[kn], n_classes, kind) if do_t2 else None
    n_t2 = len(kn)

    rows = []
    for metric in ("MCC", "AUROC", "F1", "Accuracy"):
        o = float(orig[metric])
        row = {
            "dataset": dataset, "feature": feature, "metric": metric,
            "value_original": round(o, 5),
            "n_test_original": n_test,
            "value_clean_tier1_exact": round(float(t1[metric]), 5),
            "delta_tier1": round(float(t1[metric] - o), 5),
            "n_test_tier1": n_t1,
            "value_clean_tier2_neardup": round(float(t2[metric]), 5) if t2 else np.nan,
            "delta_tier2": round(float(t2[metric] - o), 5) if t2 else np.nan,
            "n_test_tier2": n_t2 if do_t2 else np.nan,
        }
        rows.append(row)
    return rows


def run_dataset(name, k_values, n_jobs):
    train_path = os.path.join(DATA_ROOT, name, "train.csv")
    test_path = os.path.join(DATA_ROOT, name, "test.csv")
    train_seqs, y_train, test_seqs, y_test = load_dataset(train_path, test_path)
    n_classes = len(set(y_train))

    sd = _safe_ds(name)
    ke_path = os.path.join(CLEAN_DIR, f"{sd}__test_keep_exact.npy")
    kn_path = os.path.join(CLEAN_DIR, f"{sd}__test_keep_neardup.npy")
    if not os.path.exists(ke_path):
        raise SystemExit(f"Missing clean split {ke_path} (run Step 4 first).")
    keep_exact = np.load(ke_path)
    keep_neardup = np.load(kn_path) if os.path.exists(kn_path) else keep_exact
    n_test = len(test_seqs)
    y_test = np.asarray(y_test)
    print(f"\n=== {name}  (n_test={n_test}, tier1_clean={len(keep_exact)}, "
          f"tier2_clean={len(keep_neardup)}) ===")

    rows = []

    def _emit(feature, kind, model, Xte):
        rs = _rows_from(name, feature, kind, model, Xte, y_test,
                        keep_exact, keep_neardup, n_test, n_classes)
        rows.extend(rs)
        m = rs[0]  # MCC row
        t2 = "" if np.isnan(m["delta_tier2"]) else f", t2 d={m['delta_tier2']:+.4f}"
        print(f"  {feature:26s} MCC {m['value_original']:.4f} "
              f"t1={m['value_clean_tier1_exact']:.4f} (d={m['delta_tier1']:+.4f}{t2})")

    # ---- k-mer RF: FCGR once at the max grid, k-mer vectors per k ----
    max_k = max(k_values)
    grid_size = max(128, 2 ** max_k)
    g_tr = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=2)
    g_te = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=2)
    for k in k_values:
        Xtr = kmer_from_grids(g_tr, k, normalize="l1")
        Xte = kmer_from_grids(g_te, k, normalize="l1")
        model = train_rf_fast(Xtr, y_train, n_classes, n_jobs=n_jobs)
        _emit(f"kmer_k{k}_RF", "rf", model, Xte)

    # ---- FM RF + linear probe ----
    for model_name in FM_MODELS:
        tr_npz, te_npz = _find_fm(model_name, name)
        if tr_npz is None:
            print(f"  [skip] {model_name}: no cached embeddings")
            continue
        Xtr = _load_npz(tr_npz)
        Xte = _load_npz(te_npz)
        if Xtr.shape[0] != len(y_train) or Xte.shape[0] != n_test:
            print(f"  [skip] {model_name}: cache row mismatch "
                  f"(train {Xtr.shape[0]}/{len(y_train)}, test {Xte.shape[0]}/{n_test})")
            continue
        rf = train_rf_fast(Xtr, y_train, n_classes, n_jobs=n_jobs)
        _emit(f"fm_{model_name}_RF", "rf", rf, Xte)
        lp = train_linear_classifier(Xtr, y_train, n_classes, n_jobs=n_jobs)
        _emit(f"fm_{model_name}_LP", "lp", lp, Xte)

    return rows


def main():
    ap = argparse.ArgumentParser(description="E0 re-run main results on clean test")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if args.datasets:
        names = args.datasets
    elif os.path.exists(HIGH_TXT):
        names = [ln.strip() for ln in open(HIGH_TXT) if ln.strip()]
    else:
        raise SystemExit(f"No --datasets and {HIGH_TXT} missing (run Step 3).")

    all_rows = []
    for name in names:
        all_rows += run_dataset(name, args.k_values, args.n_jobs)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    # headline: MCC deltas (Tier 1 = exact/revcomp removed)
    mcc = df[df["metric"] == "MCC"]
    print("\nMCC: original -> Tier-1 clean (exact+revcomp removed):")
    print(mcc[["dataset", "feature", "value_original", "value_clean_tier1_exact",
               "delta_tier1", "n_test_original", "n_test_tier1"]].to_string(index=False))


if __name__ == "__main__":
    main()
