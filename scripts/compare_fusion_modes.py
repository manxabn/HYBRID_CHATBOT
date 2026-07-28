"""
Paired significance test: does the RRF-fusion + course-code-normalization +
alias-resolution novelty (pipeline/hybrid_retriever.py, fusion="rrf") beat
the original linear-fusion baseline, config-for-config, on the same query set?

This is the direct head-to-head test for the novelty claim -- separate from
significance_tests.py, which compares full_hybrid against the other baselines
*within* one fusion mode.

Usage (after both runs exist):
    python scripts/run_ablation.py                                   # linear (default)
    python scripts/compute_metrics.py                                # scores it
    python scripts/run_ablation.py --fusion rrf                      # rrf novelty run
    python scripts/compute_metrics.py --raw results/ablation_raw_outputs_rrf.csv \\
        --per-query-out results/ablation_metrics_per_query_rrf.csv \\
        --summary-out results/ablation_metrics_summary_rrf.csv
    python scripts/compare_fusion_modes.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LINEAR_PATH = ROOT / "results" / "ablation_metrics_per_query.csv"
RRF_PATH = ROOT / "results" / "ablation_metrics_per_query_rrf.csv"
OUT_PATH = ROOT / "results" / "fusion_mode_comparison.csv"

METRICS = ["bleu", "rougeL", "bertscore", "meteor"]
# no_retrieval doesn't touch the retriever, so fusion mode can't affect it --
# excluded as a built-in check that the two runs are otherwise comparable.
CONFIGS = ["full_hybrid", "bm25_only", "vector_only"]


def paired_test(linear_vals, rrf_vals):
    mean_diff = (rrf_vals - linear_vals).mean()
    _, p_t = stats.ttest_rel(rrf_vals, linear_vals)
    try:
        _, p_w = stats.wilcoxon(rrf_vals, linear_vals)
    except ValueError:
        p_w = float("nan")
    return mean_diff, p_t, p_w


def main():
    if not RRF_PATH.exists():
        raise SystemExit(
            f"{RRF_PATH} not found -- run the rrf ablation + compute_metrics first (see module docstring)."
        )

    lin = pd.read_csv(LINEAR_PATH)
    rrf = pd.read_csv(RRF_PATH)

    rows = []
    for config in CONFIGS:
        l = lin[lin.config == config].sort_values("query_id").reset_index(drop=True)
        r = rrf[rrf.config == config].sort_values("query_id").reset_index(drop=True)
        if len(l) != len(r) or not (l["query_id"].values == r["query_id"].values).all():
            raise SystemExit(
                f"query_id mismatch between linear and rrf runs for config={config} -- "
                "were they run against the same data/test_queries.csv?"
            )
        for metric in METRICS:
            mean_diff, p_t, p_w = paired_test(l[metric].values, r[metric].values)
            rows.append({
                "config": config,
                "metric": metric,
                "n": len(l),
                "mean_linear": round(l[metric].mean(), 4),
                "mean_rrf": round(r[metric].mean(), 4),
                "mean_diff_rrf_minus_linear": round(mean_diff, 4),
                "paired_t_p": round(p_t, 4),
                "wilcoxon_p": round(p_w, 4) if p_w == p_w else None,
                "rrf_significantly_better": bool(p_t < 0.05 and mean_diff > 0),
                "rrf_significantly_worse": bool(p_t < 0.05 and mean_diff < 0),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
