"""
Isolated significance test for adaptive routing itself (pipeline/hybrid_
retriever.py's retrieve_adaptive), distinct from the lambda-sweep evidence
that originally motivated it (paper.tex Sec.~4.3) and distinct from the
pooled adaptive_vs_full_hybrid comparison in results/ir_metrics_bootstrap_
significance.csv.

That pooled comparison (all 200 queries) is diluted by construction: for
open-ended queries, retrieve_adaptive's else-branch calls the exact same
retrieve(query, lambda_open=0.5, fusion="linear") as the full_hybrid
baseline -- there is no possible difference on that subset, only ties. Any
real effect of adaptive routing can only show up on the is_entity_heavy
subset, where adaptive takes a DIFFERENT branch (RRF fusion, lambda=0.9)
than either fixed baseline (linear fusion, lambda=0.5 or 1.0). Restricting
the paired comparison to exactly that subset is the correct isolation of
"does adaptive routing's decision help", using results/ir_metrics.csv's
already-computed per-query rows (no re-retrieval needed).

Usage: python scripts/isolate_adaptive_routing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IN_PATH = ROOT / "results" / "ir_metrics.csv"
OUT_PATH = ROOT / "results" / "adaptive_routing_isolated_significance.csv"
N_BOOTSTRAP = 2000
SEED = 42
METRICS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]


def bootstrap_ci_diff(a, b, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_approx = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), lo, hi, p_approx


def main():
    df = pd.read_csv(IN_PATH)
    entity_heavy = df[df["is_entity_heavy"]]
    n_entity_heavy = entity_heavy["query_id"].nunique()
    print(f"Entity-heavy queries: {n_entity_heavy}/{df['query_id'].nunique()}")

    configs = {c: entity_heavy[entity_heavy["config"] == c].set_index("query_id") for c in entity_heavy["config"].unique()}
    adaptive = configs["adaptive"]

    rows = []
    for other_label in ["bm25_only", "vector_only", "full_hybrid"]:
        other = configs[other_label]
        common = adaptive.index.intersection(other.index)
        for metric in METRICS:
            a = adaptive.loc[common, metric].values
            b = other.loc[common, metric].values
            # Sanity check: on this subset adaptive and the fixed baseline
            # should actually differ query-by-query for at least some rows,
            # unlike the pooled comparison where full_hybrid ties adaptive
            # on every open-ended row by construction.
            n_ties = int((a == b).sum())
            mean_diff, lo, hi, p_approx = bootstrap_ci_diff(a, b)
            rows.append({
                "comparison": f"adaptive_vs_{other_label}_entity_heavy_only", "metric": metric,
                "n": len(common), "n_ties": n_ties,
                "adaptive_mean": round(a.mean(), 4), "other_mean": round(b.mean(), 4),
                "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "p_approx": round(p_approx, 4), "significant": not (lo <= 0 <= hi),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
