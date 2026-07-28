"""
Paired bootstrap significance testing for the faithfulness (LLM-judge)
scores, per Dror, Baumer, Shlomov & Reichart (2018, ACL, "The Hitchhiker's
Guide to Testing Statistical Significance in Natural Language Processing"):
comparisons on the SAME dataset should use a paired test, and for a
graded/continuous measure like a 0-1 faithfulness score (not a binary
correct/incorrect contingency table, which is what McNemar's test assumes),
the paper's guidance is paired bootstrap or permutation testing rather than
a parametric paired t-test, since normality of the score distribution is
not automatic and the paper explicitly warns it often fails for measures
like precision/F-score -- faithfulness scores here are similarly skewed
(most answers score high, a minority low), so we default to the
non-parametric option rather than assume normality.

This closes a real, previously-flagged gap: Table~\\ref{tab:faithfulness}'s
comparison was purely descriptive (no significance test at all), because
the four baseline samples were originally treated as independent when they
in fact share the same 40 query_ids (confirmed directly against the data,
not assumed) -- a genuine paired comparison among them was always possible
and simply hadn't been run. The adaptive-pipeline sample only overlaps the
baselines' query_ids on 17/40 rows, so that comparison is restricted to
the matched subset rather than treated as if it were a full n=40/50 pair.

Usage: python scripts/bootstrap_faithfulness_significance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_BOOTSTRAP = 5000
SEED = 42
OUT_PATH = ROOT / "results" / "faithfulness_bootstrap_significance.csv"


def paired_bootstrap(a, b, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_approx = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), lo, hi, min(p_approx, 1.0)


def main():
    base = pd.read_csv(ROOT / "results" / "faithfulness_sample_baselines_faithfulness.csv")
    novel = pd.read_csv(ROOT / "results" / "faithfulness_sample_novel_faithfulness.csv")
    base = base.dropna(subset=["faithfulness"])
    novel = novel.dropna(subset=["faithfulness"])

    baseline_configs = sorted(base["config"].unique())
    rows = []

    # Among the four baselines: fully paired (same 40 query_ids each, confirmed).
    pivoted = base.pivot(index="query_id", columns="config", values="faithfulness")
    print(f"Baseline configs share {pivoted.dropna().shape[0]}/{len(pivoted)} query_ids across all four")
    for i, c1 in enumerate(baseline_configs):
        for c2 in baseline_configs[i + 1:]:
            paired = pivoted[[c1, c2]].dropna()
            mean_diff, lo, hi, p = paired_bootstrap(paired[c1], paired[c2])
            rows.append({
                "comparison": f"{c1}_vs_{c2}", "n": len(paired),
                "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "p_approx": round(p, 4), "significant": not (lo <= 0 <= hi),
            })

    # Adaptive pipeline vs each baseline: restricted to the actual overlap.
    novel_by_id = novel.set_index("query_id")["faithfulness"]
    for cfg in baseline_configs:
        base_by_id = base[base["config"] == cfg].set_index("query_id")["faithfulness"]
        shared_ids = novel_by_id.index.intersection(base_by_id.index)
        if len(shared_ids) < 5:
            print(f"Skipping adaptive_novel_vs_{cfg}: only {len(shared_ids)} shared query_ids, too few")
            continue
        mean_diff, lo, hi, p = paired_bootstrap(novel_by_id.loc[shared_ids], base_by_id.loc[shared_ids])
        rows.append({
            "comparison": f"adaptive_novel_vs_{cfg}", "n": len(shared_ids),
            "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "p_approx": round(p, 4), "significant": not (lo <= 0 <= hi),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
