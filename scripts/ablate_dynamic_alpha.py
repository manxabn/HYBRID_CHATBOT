"""
Head-to-head comparison: retrieve_adaptive (production, binary route ->
fixed {RRF@0.9, linear@0.5}) vs. retrieve_dynamic_alpha (EXPERIMENTAL,
2026-07-29, continuous per-query lambda via entity_signal_strength -- see
both methods' docstrings in pipeline/hybrid_retriever.py for the DAT-
inspired motivation and why DAT's own raw-BM25-score-dominance signal was
NOT ported as-is: a direct measurement on this corpus found it inverted --
entity_heavy queries score LOWER on raw BM25 top-1 (mean=17.8) than
open_ended ones (mean=33.0), a query-length confound, not a match-quality
signal).

Reuses scripts/measure_ir_metrics.py's own metric/relevance/bootstrap
functions directly (import, not reimplementation) so this comparison is on
identical footing (same Recall@k/MRR/nDCG definitions, same relevance
judgment, same paired-bootstrap significance procedure) as every other
retrieval-configuration comparison this project has already run.

GPU note: instantiates the real HybridRetriever (embeds queries via the
GPU-preferring Chroma1xEmbeddingFunction) -- do NOT run this concurrently
with any other GPU job (see this project's own documented resource-
contention incidents from running an embedding/Ollama job alongside GPU
fine-tuning). Retrieval-only, no Ollama/LLM generation.

Usage: python scripts/ablate_dynamic_alpha.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline.hybrid_retriever as hr
from measure_ir_metrics import QUERIES_PATH, TOP_K, run_config, bootstrap_ci_diff

OUT_PATH = ROOT / "results" / "dynamic_alpha_ablation.csv"
OUT_BOOTSTRAP = ROOT / "results" / "dynamic_alpha_bootstrap_significance.csv"


def main():
    df = pd.read_csv(QUERIES_PATH)
    retriever = hr.HybridRetriever()

    configs = {
        "adaptive": lambda r, q: r.retrieve_adaptive(q, top_n=TOP_K)[0],
        "dynamic_alpha": lambda r, q: r.retrieve_dynamic_alpha(q, top_n=TOP_K)[0],
    }

    all_results = {}
    for label, fn in configs.items():
        print(f"Running {label}...", flush=True)
        all_results[label] = run_config(retriever, df, fn, label)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)

    metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]
    summary = combined.groupby("config")[metrics].mean().round(4)
    print("\n=== Overall (n=200) ===")
    print(summary)

    print("\n=== By query type ===")
    by_type = combined.groupby(["config", "is_entity_heavy"])[metrics].mean().round(4)
    print(by_type)

    # Also report the actual entity_signal_strength/lambda distribution
    # dynamic_alpha produced, so a reader can see WHAT it did, not just the
    # resulting metrics -- e.g. whether it behaved like a near-binary router
    # in practice (strength clustering near 0 and 1) or genuinely used
    # intermediate values.
    strengths = [retriever.retrieve_dynamic_alpha(q, top_n=TOP_K)[1]["entity_signal_strength"]
                 for q in df["query"]]
    strength_counts = pd.Series(strengths).value_counts().sort_index()
    print("\n=== entity_signal_strength distribution (dynamic_alpha) ===")
    print(strength_counts)

    adaptive = all_results["adaptive"].set_index("query_id")
    dynamic = all_results["dynamic_alpha"].set_index("query_id")
    bootstrap_rows = []
    for metric in metrics:
        a = dynamic.loc[adaptive.index, metric].values
        b = adaptive[metric].values
        mean_diff, lo, hi, p_approx = bootstrap_ci_diff(a, b)
        bootstrap_rows.append({
            "comparison": "dynamic_alpha_vs_adaptive", "metric": metric,
            "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "p_approx": round(p_approx, 4), "significant": not (lo <= 0 <= hi),
        })
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(OUT_BOOTSTRAP, index=False)
    print("\n=== Paired bootstrap significance (dynamic_alpha vs. adaptive, 2000 resamples) ===")
    print(bootstrap_df.to_string(index=False))
    print(f"\nWrote {OUT_PATH} and {OUT_BOOTSTRAP}")
    print("\nNOTE: dynamic_alpha is EXPERIMENTAL. Only adopt it over retrieve_adaptive "
          "if this comparison shows a significant, non-negative improvement on the metrics "
          "that matter (recall@1/mrr especially) -- a null or negative result here is a real, "
          "reportable finding (DAT's mechanism, adapted, didn't transfer), not a failure to hide.")


if __name__ == "__main__":
    main()
