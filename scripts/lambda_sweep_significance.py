"""
Paired significance tests within the lambda sweep: is lambda=0.5 (or any
other point) significantly different from the lambda=1.0 (BM25-only) and
lambda=0.0 (vector-only) endpoints? Answers whether the metric-vs-lambda
plot's apparent mid-range peak is real or sampling noise.
"""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PER_QUERY_PATH = ROOT / "results" / "lambda_sweep_metrics_per_query.csv"
OUT_PATH = ROOT / "results" / "lambda_sweep_significance.csv"

METRICS = ["bleu", "rougeL", "bertscore", "meteor"]
LAMBDAS = [round(i * 0.1, 1) for i in range(11)]


def paired(df, metric, lam_a, lam_b):
    da = df[df.lam == lam_a].sort_values("query_id")[metric].values
    db = df[df.lam == lam_b].sort_values("query_id")[metric].values
    mean_diff = (da - db).mean()
    _, p = stats.ttest_rel(da, db)
    return mean_diff, p


def main():
    df = pd.read_csv(PER_QUERY_PATH)
    df.rename(columns={"lambda": "lam"}, inplace=True)

    rows = []
    for subset_name, subset in [("all", df), ("entity_heavy", df[df.is_entity_heavy]),
                                 ("open_ended", df[~df.is_entity_heavy])]:
        for lam in LAMBDAS:
            if lam in (0.0, 1.0):
                continue
            for endpoint, label in [(1.0, "vs_bm25_only"), (0.0, "vs_vector_only")]:
                for metric in METRICS:
                    mean_diff, p = paired(subset, metric, lam, endpoint)
                    rows.append({
                        "subset": subset_name,
                        "lambda": lam,
                        "comparison": label,
                        "metric": metric,
                        "n": len(subset[subset.lam == lam]),
                        "mean_diff": round(mean_diff, 4),
                        "p_value": round(p, 4),
                        "significant_at_0.05": bool(p < 0.05),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(f"Significant results: {out['significant_at_0.05'].sum()} / {len(out)}")
    print(out[out["significant_at_0.05"]].to_string(index=False))


if __name__ == "__main__":
    main()
