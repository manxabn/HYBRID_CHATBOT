"""
Paired significance tests (paired t-test + Wilcoxon signed-rank) between
full_hybrid and each single-method baseline, on all four metrics, overall
and on the entity-heavy subset. Answers: is the gap between configs real,
or noise at n=200 (n=100 for the entity-heavy subset)?
"""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PER_QUERY_PATH = ROOT / "results" / "ablation_metrics_per_query.csv"
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "significance_tests.csv"

METRICS = ["bleu", "rougeL", "bertscore", "meteor"]
BASELINES = ["bm25_only", "vector_only", "no_retrieval"]


def paired_test(df, metric, a, b):
    da = df[df.config == a].sort_values("query_id")[metric].values
    db = df[df.config == b].sort_values("query_id")[metric].values
    mean_diff = (da - db).mean()
    _, p_t = stats.ttest_rel(da, db)
    try:
        _, p_w = stats.wilcoxon(da, db)
    except ValueError:
        p_w = float("nan")
    return mean_diff, p_t, p_w


def main():
    df = pd.read_csv(PER_QUERY_PATH)
    tq = pd.read_csv(QUERIES_PATH)[["query_id", "is_entity_heavy"]]
    df = df.merge(tq, on="query_id")

    rows = []
    for subset_name, subset in [("all", df), ("entity_heavy", df[df.is_entity_heavy]),
                                 ("open_ended", df[~df.is_entity_heavy])]:
        for baseline in BASELINES:
            for metric in METRICS:
                mean_diff, p_t, p_w = paired_test(subset, metric, "full_hybrid", baseline)
                rows.append({
                    "subset": subset_name,
                    "comparison": f"full_hybrid_vs_{baseline}",
                    "metric": metric,
                    "n": len(subset[subset.config == "full_hybrid"]),
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
