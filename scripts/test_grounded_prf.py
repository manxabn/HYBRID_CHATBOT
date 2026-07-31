"""
A new, custom-built retrieval variant, not from any single paper -- designed
specifically for this corpus's own failure mode, not a generic technique
applied blindly. Motivation: HyDE (already tested, results/hyde_retrieval_
raw.csv) failed because it expands the query with an LLM-IMAGINED
hypothetical answer, which can be flatly wrong since the LLM doesn't
actually know this corpus's real facts -- confirmed as the failure
mechanism there.

This variant, "grounded pseudo-relevance feedback": do a first-pass
retrieval with the REAL corpus (no LLM involved), take the actual top-1
retrieved chunk's real text, and use ITS embedding blended with the
original query's embedding for a second-pass vector search -- grounding
the "expansion" in real corpus content instead of imagined content. This
is conceptually related to classic pseudo-relevance feedback (RM3) and to
ANCE-PRF, but implemented directly against this project's own retriever
rather than adopting either paper's exact recipe unmodified, since this
corpus's structure (mostly single-fact chunks, no long documents) doesn't
match what either technique was designed for. Reported as an experiment,
not assumed to work.

Blend: new_vector = normalize(alpha * query_vec + (1-alpha) * top1_vec),
alpha=0.7 (query dominates, top-1 chunk nudges the search direction
toward where real relevant content already is -- if the first-pass top-1
is wrong, a small nudge does limited damage; if it's right, it should
reinforce finding more of the same real, correct content).

Retrieval-only (no Ollama needed -- the "expansion" reuses a real
retrieved chunk's already-computed embedding, no generation involved).
Open-ended queries only, same reasoning as the HyDE test (entity-heavy is
saturated by exact-match regardless of vector quality).

Usage: python scripts/test_grounded_prf.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from scripts.measure_ir_metrics import is_relevant, recall_at_k, rr, ndcg_at_k, TOP_K

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "grounded_prf_raw.csv"
ALPHA = 0.7


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == False].reset_index(drop=True)
    print(f"Open-ended queries: {len(df)}", flush=True)

    retriever = hr.HybridRetriever()

    rows = []
    for _, row in df.iterrows():
        query = row["query"]
        bm25_cand = retriever._bm25_candidates(query)
        exact_match_ids = retriever._exact_match_ids(query)

        query_embedding = np.array(retriever.embedding_function.embed_query(query))
        vec_cand_query = retriever._vector_candidates(query, query_embedding=query_embedding.tolist())

        # First pass: get the real top-1 chunk's own embedding (already
        # computed and stored by Chroma at index time -- re-embed its text
        # directly here for simplicity/correctness, avoids needing a
        # separate Chroma lookup-by-id call).
        scored_first = retriever._score_linear(bm25_cand, vec_cand_query, exact_match_ids, 0.0)
        scored_first.sort(key=lambda x: x["score"], reverse=True)
        top1_text = scored_first[0]["text"] if scored_first else ""
        top1_embedding = np.array(retriever.embedding_function.embed_query(top1_text)) if top1_text else query_embedding

        blended = ALPHA * query_embedding + (1 - ALPHA) * top1_embedding
        blended = blended / (np.linalg.norm(blended) + 1e-8)
        vec_cand_prf = retriever._vector_candidates(query, query_embedding=blended.tolist())

        for label, vec_cand, lam in [
            ("vector_only_baseline", vec_cand_query, 0.0),
            ("vector_only_grounded_prf", vec_cand_prf, 0.0),
            ("full_hybrid_baseline", vec_cand_query, 0.5),
            ("full_hybrid_grounded_prf", vec_cand_prf, 0.5),
            ("bm25_only", vec_cand_query, 1.0),
        ]:
            scored = retriever._score_linear(bm25_cand, vec_cand, exact_match_ids, lam)
            scored.sort(key=lambda x: x["score"], reverse=True)
            results = scored[:TOP_K]
            rel_ranks = [i for i, c in enumerate(results) if is_relevant(c, query, str(row["reference_answer"]), False)]
            rows.append({
                "query_id": row["query_id"], "config": label,
                "recall@1": recall_at_k(rel_ranks, 1), "recall@3": recall_at_k(rel_ranks, 3),
                "recall@5": recall_at_k(rel_ranks, 5), "mrr": rr(rel_ranks),
                "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print("\n=== Recall by config (open-ended, n=100) ===")
    print(out.groupby("config")[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
