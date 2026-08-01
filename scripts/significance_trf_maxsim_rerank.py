"""
Paired significance test for scripts/test_trf_maxsim_rerank.py's raw
output (full_hybrid_baseline vs full_hybrid_trf_rerank, same 100
open-ended queries, matched by query_id).

Usage: python scripts/significance_trf_maxsim_rerank.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / "results" / "trf_maxsim_rerank_raw.csv"
OUT_PATH = ROOT / "results" / "trf_maxsim_rerank_significance.csv"
METRICS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]


def main():
    df = pd.read_csv(RAW_PATH)
    base = df[df["config"] == "full_hybrid_baseline"].set_index("query_id")
    trf = df[df["config"] == "full_hybrid_trf_rerank"].set_index("query_id")
    common = base.index.intersection(trf.index)
    print(f"n={len(common)}")

    rows = []
    for metric in METRICS:
        b = base.loc[common, metric].values
        t = trf.loc[common, metric].values
        diffs = t - b
        n_changed = int((diffs != 0).sum())
        if diffs.std() == 0:
            rows.append({"metric": metric, "mean_diff": 0.0, "n_changed": n_changed,
                         "paired_t_p": None, "wilcoxon_p": None, "significant": False})
            print(f"{metric}: no variance, n_changed={n_changed}")
            continue
        t_stat, p_t = stats.ttest_rel(t, b)
        try:
            w_stat, p_w = stats.wilcoxon(t, b)
        except ValueError:
            p_w = float("nan")
        sig = bool(p_t < 0.05 and p_w < 0.05)
        rows.append({"metric": metric, "mean_diff": round(float(diffs.mean()), 4),
                     "n_changed": n_changed, "paired_t_p": round(float(p_t), 4),
                     "wilcoxon_p": round(float(p_w), 4), "significant": sig})
        print(f"{metric}: mean_diff={diffs.mean():.4f} n_changed={n_changed} "
              f"paired_t_p={p_t:.4f} wilcoxon_p={p_w:.4f} significant={sig}")

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
