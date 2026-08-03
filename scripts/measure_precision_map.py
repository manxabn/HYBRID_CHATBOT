"""
Precision@k and MAP -- two retrieval metrics named in an external "A*
evaluation framework" checklist that this project's existing IR-metrics
script (measure_ir_metrics.py, which already computes Recall@k/MRR/
nDCG@5/@10) does not separately report. Reuses that script's own
is_relevant() relevance judgment directly (imported, not reimplemented)
so these two new metrics are judged by the EXACT same relevance
definition already used and reported for Recall/MRR/nDCG -- not a
different, inconsistent methodology.

Same 200-query English test set, same four configs (bm25_only,
vector_only, full_hybrid, adaptive), same TOP_K=10 candidate window.

Precision@k = (# relevant docs in top-k) / k.
MAP = mean, over queries, of average precision (the mean of
precision-at-rank for each relevant doc's rank, using the same
multi-relevant-doc-aware rel_ranks list measure_ir_metrics.py's own
nDCG@k already accounts for -- this corpus has genuine near-duplicate/
paraphrase content where R>1 relevant chunks per query is common, not
an edge case, per that script's own module docstring).

Usage: python scripts/measure_precision_map.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_ir_metrics as irm

OUT_PATH = ROOT / "results_final" / "retrieval" / "precision_map.csv"
K_VALUES = [1, 3, 5, 10]


def precision_at_k(rel_ranks, k):
    return len([r for r in rel_ranks if r < k]) / k


def average_precision(rel_ranks, k):
    """Mean of precision@rank for each relevant doc found within top-k,
    0.0 if none found -- standard AP@k, consistent with nDCG@k's own
    multi-relevant-doc handling in measure_ir_metrics.py."""
    hits = sorted(r for r in rel_ranks if r < k)
    if not hits:
        return 0.0
    precisions = [(i + 1) / (rank + 1) for i, rank in enumerate(hits)]
    return sum(precisions) / len(hits)


def main():
    df = pd.read_csv(irm.QUERIES_PATH)
    retriever = irm.hr.HybridRetriever()

    configs = {
        "bm25_only": lambda r, q: r.retrieve(q, 1.0, fusion="linear", top_n=irm.TOP_K),
        "vector_only": lambda r, q: r.retrieve(q, 0.0, fusion="linear", top_n=irm.TOP_K),
        "full_hybrid": lambda r, q: r.retrieve(q, 0.5, fusion="linear", top_n=irm.TOP_K),
        "adaptive": lambda r, q: r.retrieve_adaptive(q, top_n=irm.TOP_K)[0],
    }

    rows = []
    for label, fn in configs.items():
        print(f"Running {label}...", flush=True)
        for _, r in df.iterrows():
            results = fn(retriever, r["query"])
            rel_ranks = [i for i, c in enumerate(results[:irm.TOP_K])
                         if irm.is_relevant(c, r["query"], str(r["reference_answer"]), r["is_entity_heavy"])]
            row = {"query_id": r["query_id"], "config": label, "is_entity_heavy": r["is_entity_heavy"]}
            for k in K_VALUES:
                row[f"precision@{k}"] = precision_at_k(rel_ranks, k)
            row["ap"] = average_precision(rel_ranks, irm.TOP_K)
            rows.append(row)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    precision_cols = [f"precision@{k}" for k in K_VALUES]
    summary = out.groupby("config")[precision_cols + ["ap"]].mean().round(4)
    summary = summary.rename(columns={"ap": "MAP"})
    print("\n=== Precision@k and MAP (n=200) ===")
    print(summary)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
