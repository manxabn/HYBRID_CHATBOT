"""
Paired significance test (paired t-test + Wilcoxon, both required to agree
per this project's convention -- see scripts/significance_tests_novel.py,
whose paired_test() this reuses verbatim) for the deployed adaptive
pipeline vs. the ColBERT-v2-as-retriever end-to-end generation-quality
ablation (scripts/colbert_generate_and_score.py + compute_metrics.py).

Matched on the subset of query_ids the adaptive pipeline did not abstain
on, same fairness rule as significance_tests_novel.py.

Usage: python scripts/significance_adaptive_vs_colbert_generation.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ADAPTIVE_PATH = ROOT / "results" / "novel_pipeline_metrics_per_query_roundO_noreranker.csv"
COLBERT_PATH = ROOT / "results" / "colbert_external_generation_metrics_per_query.csv"
OUT_PATH = ROOT / "results" / "significance_adaptive_vs_colbert_generation.csv"
METRICS = ["bleu", "rougeL", "bertscore", "meteor"]


def paired_test(a_vals, b_vals):
    mean_diff = (a_vals - b_vals).mean()
    _, p_t = stats.ttest_rel(a_vals, b_vals)
    try:
        _, p_w = stats.wilcoxon(a_vals, b_vals)
    except ValueError:
        p_w = float("nan")
    return mean_diff, p_t, p_w


def main():
    adaptive = pd.read_csv(ADAPTIVE_PATH)
    colbert = pd.read_csv(COLBERT_PATH)

    n_total = len(adaptive)
    abstained_col = adaptive["abstained"].astype(str)
    non_abstained = adaptive[abstained_col == "False"]
    print(f"Adaptive pipeline: {n_total} total, {len(non_abstained)} answered "
          f"({n_total - len(non_abstained)} abstained)")

    matched_ids = set(non_abstained["query_id"]) & set(colbert["query_id"])
    adaptive_subset = non_abstained[non_abstained["query_id"].isin(matched_ids)].sort_values("query_id")
    colbert_subset = colbert[colbert["query_id"].isin(matched_ids)].sort_values("query_id")
    assert list(adaptive_subset["query_id"]) == list(colbert_subset["query_id"]), "query_id mismatch"
    print(f"Matched n={len(matched_ids)}")

    rows = []
    for metric in METRICS:
        mean_diff, p_t, p_w = paired_test(adaptive_subset[metric].values, colbert_subset[metric].values)
        rows.append({
            "comparison": "adaptive_vs_colbert_external_generation", "metric": metric,
            "n_matched": len(matched_ids),
            "adaptive_mean": round(adaptive_subset[metric].mean(), 4),
            "colbert_external_mean": round(colbert_subset[metric].mean(), 4),
            "mean_diff": round(mean_diff, 4),
            "paired_t_p": round(p_t, 4),
            "wilcoxon_p": round(p_w, 4) if p_w == p_w else None,
            "significant_both_tests": bool(p_t < 0.05 and p_w == p_w and p_w < 0.05),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
