"""
Paired significance test for the confidence-ordered "zig-zag" context
assembly (pipeline/novel_pipeline.py's use_confidence_ordering flag) --
the direct follow-up to scripts/ablate_confidence_ordering.py, which only
generates and writes raw answers (results/confidence_ordering_ablation_raw.
csv) without scoring them. Scored separately via scripts/compute_metrics.py
(results/confidence_ordering_metrics_per_query.csv), same BLEU/ROUGE-L/
BERTScore/METEOR as every other generation-quality ablation in this
project. This script does the paired comparison, same method as
scripts/significance_tests.py (paired t-test + Wilcoxon signed-rank).

Usage: python scripts/confidence_ordering_significance.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PER_QUERY_PATH = ROOT / "results" / "confidence_ordering_metrics_per_query.csv"
OUT_PATH = ROOT / "results" / "confidence_ordering_significance.csv"

METRICS = ["bleu", "rougeL", "bertscore", "meteor"]


def paired_test(df, metric, a, b):
    da = df[df.config == a].sort_values("query_id")[metric].values
    db = df[df.config == b].sort_values("query_id")[metric].values
    ids_a = df[df.config == a].sort_values("query_id")["query_id"].values
    ids_b = df[df.config == b].sort_values("query_id")["query_id"].values
    assert (ids_a == ids_b).all(), "query_id mismatch between ordering_on and ordering_off rows"
    mean_diff = (da - db).mean()
    _, p_t = stats.ttest_rel(da, db)
    try:
        _, p_w = stats.wilcoxon(da, db)
    except ValueError:
        p_w = float("nan")
    return mean_diff, p_t, p_w, len(da)


def main():
    df = pd.read_csv(PER_QUERY_PATH)
    rows = []
    for metric in METRICS:
        mean_diff, p_t, p_w, n = paired_test(df, metric, "ordering_on", "ordering_off")
        rows.append({
            "comparison": "ordering_on_vs_ordering_off", "metric": metric, "n": n,
            "mean_diff": round(mean_diff, 4),
            "paired_t_p": round(p_t, 4),
            "wilcoxon_p": round(p_w, 4) if p_w == p_w else None,
            "significant_at_0.05": bool(p_t < 0.05),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
