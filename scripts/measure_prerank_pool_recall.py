"""
Diagnostic for WHY the cross-encoder reranker hurts this pipeline (Section
"Reranker Ablation"): measures Recall@10 of the adaptive retriever's
candidate pool BEFORE reranking (rerank_pool_size=10, pipeline/novel_
pipeline.py default) -- i.e., does the reranker even have room to help?

A reranker can only ever help if the correct chunk is somewhere in its
input candidate pool but not already ranked first; if the base retriever's
pool already has the correct chunk at rank 1 in the overwhelming majority
of cases (near-ceiling recall@10 AND recall@1), there is structurally
nothing left for a reranker to fix, and adding one can only ever inject
noise, not recover missed candidates -- a mechanistic, directly-testable
explanation for a negative reranker result, independent of any reranker
quality question.

Retrieval-only, no LLM generation, no reranker actually invoked -- this
measures the retriever's own output, the same is_relevant/recall_at_k
logic already used by scripts/measure_ir_metrics.py (reused directly, not
reimplemented, so this is consistent with every other reported recall
number in this project).

Usage: python scripts/measure_prerank_pool_recall.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline.hybrid_retriever as hr
from measure_ir_metrics import is_relevant, recall_at_k, rr

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "prerank_pool_recall.csv"
POOL_SIZE = 10  # pipeline/novel_pipeline.py's rerank_pool_size default


def main():
    df = pd.read_csv(QUERIES_PATH)
    retriever = hr.HybridRetriever()

    rows = []
    for _, r in df.iterrows():
        results, _ = retriever.retrieve_adaptive(r["query"], top_n=POOL_SIZE)
        rel_ranks = [i for i, c in enumerate(results[:POOL_SIZE])
                     if is_relevant(c, r["query"], str(r["reference_answer"]), r["is_entity_heavy"])]
        rows.append({
            "query_id": r["query_id"], "is_entity_heavy": r["is_entity_heavy"],
            "recall@1": recall_at_k(rel_ranks, 1),
            "recall@5": recall_at_k(rel_ranks, 5),
            f"recall@{POOL_SIZE}": recall_at_k(rel_ranks, POOL_SIZE),
            "mrr": rr(rel_ranks),
            "correct_chunk_present_but_not_rank1": (recall_at_k(rel_ranks, POOL_SIZE) == 1.0
                                                     and recall_at_k(rel_ranks, 1) == 0.0),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print(f"n={len(out)}")
    print(f"Recall@1 (already correct before reranking): {out['recall@1'].mean():.4f}")
    print(f"Recall@5: {out['recall@5'].mean():.4f}")
    print(f"Recall@{POOL_SIZE} (ceiling reranking could ever reach): {out[f'recall@{POOL_SIZE}'].mean():.4f}")
    print(f"MRR: {out['mrr'].mean():.4f}")
    n_fixable = out["correct_chunk_present_but_not_rank1"].sum()
    print(f"\nQueries where the correct chunk IS in the pool but NOT already rank-1 "
          f"(the only queries a reranker could possibly improve): {n_fixable}/{len(out)} "
          f"({n_fixable / len(out):.1%})")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
