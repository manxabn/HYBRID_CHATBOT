"""
Sweep RRF_K (pipeline/hybrid_retriever.py's RRF_K=60, fixed since RRF fusion
was first implemented and never tuned for this corpus -- flagged directly in
paper.tex as an open item) on the 100 entity-heavy test queries, i.e. exactly
the subset that actually uses RRF fusion via retrieve_adaptive.

Motivated by scripts/isolate_adaptive_routing.py's finding: on entity-heavy
queries with an unambiguous exact match, the UNAMBIGUOUS_MATCH_SCORE bonus
and the forced bm25_rank=0 already fix the top-ranked document regardless of
RRF_K, so we expect (and explicitly check) that RRF_K only has room to matter
on the minority of entity-heavy queries WITHOUT an unambiguous exact match --
reporting both the overall sweep and that subset split, rather than only the
pooled number, so a null result on the full set isn't misread as "RRF_K
doesn't matter" when it may simply have no queries left where it could.

Retrieval-only, no LLM generation -- safe alongside a concurrently running
Ollama job.

Usage: python scripts/sweep_rrf_k.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "rrf_k_sweep.csv"
TOP_K = 10
K_VALUES = [10, 20, 40, 60, 100, 200]


def ndcg_at_k(rel_ranks, k):
    dcg = sum(1.0 / np.log2(r + 2) for r in rel_ranks if r < k)
    n_relevant = min(len(rel_ranks), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(rel_ranks, k):
    return 1.0 if any(r < k for r in rel_ranks) else 0.0


def rr(rel_ranks):
    return 1.0 / (min(rel_ranks) + 1) if rel_ranks else 0.0


def is_relevant(candidate, query, reference_answer):
    # Reuses the same entity-heavy relevance logic as measure_ir_metrics.py.
    from measure_ir_metrics import is_relevant as _is_relevant
    return _is_relevant(candidate, query, reference_answer, True)


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    df = pd.read_csv(QUERIES_PATH)
    entity_heavy = df[df["is_entity_heavy"]].reset_index(drop=True)
    print(f"Entity-heavy queries: {len(entity_heavy)}/{len(df)}")

    retriever = hr.HybridRetriever()

    # Split by whether the query has an unambiguous exact match (len==1),
    # since that's exactly the condition that makes RRF_K irrelevant.
    has_unambiguous = []
    for _, r in entity_heavy.iterrows():
        ids = retriever._exact_match_ids(r["query"])
        has_unambiguous.append(len(ids) == 1)
    entity_heavy["has_unambiguous_match"] = has_unambiguous
    n_unambig = sum(has_unambiguous)
    print(f"Of these, {n_unambig}/{len(entity_heavy)} have an unambiguous (single) exact match; "
          f"{len(entity_heavy) - n_unambig} do not, and are the only ones RRF_K can affect.")

    rows = []
    for k in K_VALUES:
        hr.RRF_K = k
        for _, r in entity_heavy.iterrows():
            results = retriever.retrieve(r["query"], 0.9, fusion="rrf", top_n=TOP_K)
            rel_ranks = [i for i, c in enumerate(results[:TOP_K])
                         if is_relevant(c, r["query"], str(r["reference_answer"]))]
            rows.append({
                "rrf_k": k, "query_id": r["query_id"],
                "has_unambiguous_match": r["has_unambiguous_match"],
                "recall@1": recall_at_k(rel_ranks, 1), "recall@5": recall_at_k(rel_ranks, 5),
                "mrr": rr(rel_ranks), "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
            })
        print(f"  rrf_k={k} done", flush=True)
    hr.RRF_K = 60  # restore default

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("\n=== Overall (all 100 entity-heavy queries) ===")
    print(out.groupby("rrf_k")[["recall@1", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4))

    print("\n=== Only queries WITHOUT an unambiguous exact match (n={}) ===".format(len(entity_heavy) - n_unambig))
    no_unambig = out[~out["has_unambiguous_match"]]
    if len(no_unambig):
        print(no_unambig.groupby("rrf_k")[["recall@1", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4))
    else:
        print("(none -- every entity-heavy test query has an unambiguous exact match)")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
