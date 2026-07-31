"""
HyDE (Hypothetical Document Embeddings, Gao et al. 2022): instead of
embedding the raw query for vector search, ask the LLM to generate a
short hypothetical answer first, then embed THAT and search with it.
Motivation: a bare question ("What amenities does TARC offer?") may sit
further in embedding space from the corpus's actual answer text than a
plausible hypothetical answer would ("TARC offers residential facilities,
common rooms, ...") -- closing that gap could specifically strengthen
hybrid's vector component on open-ended queries, a genuinely different
lever from every fusion-weight/reranker/BM25-tuning attempt tried so far
this session.

Retrieval-only (recall@k), not full generation -- one short Ollama call
per query to produce the hypothetical answer, then real BM25+vector
retrieval exactly as pipeline/hybrid_retriever.py already implements it,
just with the HyDE embedding substituted for the query embedding on the
vector side. BM25 side is unchanged (still searches the real query text
-- HyDE only ever applies to dense retrieval in the original paper).

Scoped to the 100 open-ended queries specifically: entity-heavy queries
are already saturated by the exact-match mechanism regardless of vector
quality (established repeatedly this session), so this is the one subset
where a real embedding-space improvement could actually show through.

Usage: python scripts/test_hyde_retrieval.py
"""

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from pipeline.ollama_client import generate
from scripts.measure_ir_metrics import is_relevant, recall_at_k, rr, ndcg_at_k, TOP_K

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_RAW = ROOT / "results" / "hyde_retrieval_raw.csv"
OUT_HYPOTHETICALS = ROOT / "results" / "hyde_hypothetical_answers.csv"

HYDE_PROMPT_TEMPLATE = (
    "Write a short, plausible-sounding factual answer to this question about "
    "BRAC University academic advising, even if you are not certain it is "
    "correct. One or two sentences only, no caveats or hedging.\n\nQuestion: {q}"
)


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == False].reset_index(drop=True)
    print(f"Open-ended queries: {len(df)}", flush=True)

    retriever = hr.HybridRetriever()

    # Resume support for the hypothetical-answer generation step (Ollama-bound).
    completed = {}
    if OUT_HYPOTHETICALS.exists() and OUT_HYPOTHETICALS.stat().st_size > 0:
        prior = pd.read_csv(OUT_HYPOTHETICALS)
        completed = dict(zip(prior["query_id"], prior["hypothetical_answer"]))
        print(f"Resuming: {len(completed)} hypothetical answers already generated", flush=True)

    write_header = not (OUT_HYPOTHETICALS.exists() and OUT_HYPOTHETICALS.stat().st_size > 0)
    with open(OUT_HYPOTHETICALS, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "hypothetical_answer"])
        if write_header:
            writer.writeheader()
            f.flush()
        for _, row in df.iterrows():
            if row["query_id"] in completed:
                continue
            prompt = HYDE_PROMPT_TEMPLATE.format(q=row["query"])
            hyp = generate(prompt, context=None)
            completed[row["query_id"]] = hyp
            writer.writerow({"query_id": row["query_id"], "query": row["query"], "hypothetical_answer": hyp})
            f.flush()
            print(f"  {row['query_id']}: {hyp[:70]!r}", flush=True)

    rows = []
    for _, row in df.iterrows():
        query = row["query"]
        hyp_answer = completed[row["query_id"]]

        bm25_cand = retriever._bm25_candidates(query)
        exact_match_ids = retriever._exact_match_ids(query)

        # Baseline: normal query embedding for vector side.
        query_embedding = retriever.embedding_function.embed_query(query)
        vec_cand_query = retriever._vector_candidates(query, query_embedding=query_embedding)
        # HyDE: hypothetical-answer embedding for vector side instead.
        hyde_embedding = retriever.embedding_function.embed_query(hyp_answer)
        vec_cand_hyde = retriever._vector_candidates(query, query_embedding=hyde_embedding)

        for label, vec_cand in [("vector_only_baseline", vec_cand_query), ("vector_only_hyde", vec_cand_hyde)]:
            scored = retriever._score_linear(bm25_cand, vec_cand, exact_match_ids, 0.0)
            scored.sort(key=lambda x: x["score"], reverse=True)
            results = scored[:TOP_K]
            rel_ranks = [i for i, c in enumerate(results) if is_relevant(c, query, str(row["reference_answer"]), False)]
            rows.append({
                "query_id": row["query_id"], "config": label,
                "recall@1": recall_at_k(rel_ranks, 1), "recall@3": recall_at_k(rel_ranks, 3),
                "recall@5": recall_at_k(rel_ranks, 5), "mrr": rr(rel_ranks),
                "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
            })

        for label, vec_cand, lam in [("bm25_only", vec_cand_query, 1.0)]:
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

        for label, vec_cand in [("full_hybrid_baseline", vec_cand_query), ("full_hybrid_hyde", vec_cand_hyde)]:
            scored = retriever._score_linear(bm25_cand, vec_cand, exact_match_ids, 0.5)
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
    out.to_csv(OUT_RAW, index=False)
    print("\n=== Recall by config (open-ended, n=100) ===")
    print(out.groupby("config")[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4))
    print(f"\nWrote {OUT_RAW}")


if __name__ == "__main__":
    main()
