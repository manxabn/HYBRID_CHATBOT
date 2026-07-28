"""
Paired significance tests: adaptive_novel pipeline vs each of the 4 baseline
configs, on the matched subset of query_ids the novel pipeline actually
answered (did not abstain on) -- comparing a system that can abstain against
systems that never abstain is only fair on the subset where both produced a
real answer. The novel pipeline's abstention rate/coverage is reported
separately, not folded into the same average, since it's a distinct
mechanism (trading coverage for precision), not a quality score.

Usage:
    python scripts/significance_tests_novel.py \
        --novel results/novel_pipeline_metrics_per_query_roundB.csv \
        --baselines results/ablation_metrics_per_query_roundB.csv \
        --out results/significance_tests_novel_roundB.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["bleu", "rougeL", "bertscore", "meteor"]
BASELINES = ["full_hybrid", "bm25_only", "vector_only", "no_retrieval"]


def paired_test(novel_vals, base_vals):
    mean_diff = (novel_vals - base_vals).mean()
    _, p_t = stats.ttest_rel(novel_vals, base_vals)
    try:
        _, p_w = stats.wilcoxon(novel_vals, base_vals)
    except ValueError:
        p_w = float("nan")
    return mean_diff, p_t, p_w


def main(novel_path: Path, baselines_path: Path, out_path: Path):
    novel = pd.read_csv(novel_path)
    baselines = pd.read_csv(baselines_path)

    n_total = len(novel)
    # pandas reads the CSV's "True"/"False" strings as strings, not bool --
    # compare against the string form rather than Python False (which would
    # silently match nothing and make it look like abstention never fires).
    abstained_col = novel["abstained"].astype(str)
    non_abstained = novel[abstained_col == "False"]
    n_answered = len(non_abstained)
    print(f"Novel pipeline: {n_total} total queries, {n_answered} answered "
          f"({n_total - n_answered} abstained, "
          f"{100*(n_total-n_answered)/n_total:.1f}% abstention rate)")

    matched_ids = set(non_abstained["query_id"])

    rows = []
    for baseline in BASELINES:
        base_subset = baselines[(baselines["config"] == baseline)
                                 & (baselines["query_id"].isin(matched_ids))]
        base_subset = base_subset.sort_values("query_id")
        novel_subset = non_abstained.sort_values("query_id")
        assert list(base_subset["query_id"]) == list(novel_subset["query_id"]), \
            f"query_id mismatch between novel and {baseline} on matched subset"

        for metric in METRICS:
            mean_diff, p_t, p_w = paired_test(
                novel_subset[metric].values, base_subset[metric].values)
            rows.append({
                "comparison": f"adaptive_novel_vs_{baseline}",
                "metric": metric,
                "n_matched": len(base_subset),
                "novel_mean": round(novel_subset[metric].mean(), 4),
                "baseline_mean": round(base_subset[metric].mean(), 4),
                "mean_diff": round(mean_diff, 4),
                "paired_t_p": round(p_t, 4),
                "wilcoxon_p": round(p_w, 4) if p_w == p_w else None,
                "significant_at_0.05": bool(p_t < 0.05),
            })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.novel, args.baselines, args.out)
