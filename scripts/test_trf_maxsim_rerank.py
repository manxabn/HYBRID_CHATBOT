"""
Tests Tensor-based Re-ranking Fusion (TRF), a technique flagged in this
project's own literature review (Wang et al., "Balancing the Blend: An
Experimental Analysis of Trade-offs in Hybrid Search", arXiv:2508.01405,
2025 -- independently verified via WebFetch of the arXiv HTML before
implementation, not taken from a one-line paraphrase) as beating RRF by
+8.1% nDCG@10 on a hybrid full-text+dense benchmark.

IMPORTANT SCOPE DISCLOSURE, verified before writing any code: TRF is NOT
a simple score-fusion formula like RRF or linear blending. It is a
second-stage re-ranker built on the MaxSim operation from late-interaction
("tensor search") models like ColBERT --
    sim(Q,D) = sum_i max_j (q_i . d_j)
over PER-TOKEN query/document embeddings, not single pooled vectors. The
paper's TRF results were measured against a purpose-built multi-vector
retrieval model. This project has no such model deployed, and training
one from scratch is out of scope for a single experiment.

What this script actually tests, honestly scoped: whether MaxSim
re-ranking helps AT ALL when computed against this project's own already
-deployed sentence-embedding model's raw per-token hidden states (via
sentence-transformers' output_value="token_embeddings", no new model
download or training needed) -- the same "implemented directly against
this project's own retriever rather than adopting either paper's exact
recipe unmodified" pattern already used for the grounded-PRF experiment
(scripts/test_grounded_prf.py). This is NOT a faithful reproduction of
TRF's reported benchmark result: all-MiniLM-L6-v2 (and this project's
fine-tuned variants of it) was trained with mean-pooling as its objective,
never trained for token-level late-interaction scoring, so a negative
result here would not disprove TRF -- it would only show that this
specific corpus's off-the-shelf token embeddings aren't suited to MaxSim
without dedicated training. Reported as exactly that scope, not oversold.

Method: for each open-ended query (entity-heavy is saturated by exact
-match regardless of any other signal, same reasoning as every other
retrieval-lever experiment tonight), take the full_hybrid candidate pool
(top 10), re-rank it by MaxSim between the query's own per-token
embeddings and each candidate's per-token embeddings (both L2-normalized
per token, matching the ColBERT/TRF convention so the dot product is a
cosine similarity), and compare recall@1/3/5, MRR, nDCG@5/@10 against the
original full_hybrid/bm25_only/vector_only ranking.

Retrieval-only (encoding + MaxSim compute, no Ollama needed).

Usage: python scripts/test_trf_maxsim_rerank.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
import pipeline.chroma_embedding as ce
from scripts.measure_ir_metrics import is_relevant, recall_at_k, rr, ndcg_at_k, TOP_K

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "trf_maxsim_rerank_raw.csv"


def token_embeddings(model, text: str) -> torch.Tensor:
    """Per-token embeddings, L2-normalized per token (ColBERT/TRF
    convention -- makes the dot product a per-token cosine similarity)."""
    emb = model.encode(text, output_value="token_embeddings", convert_to_tensor=True)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def maxsim(q_emb: torch.Tensor, d_emb: torch.Tensor) -> float:
    """sim(Q,D) = sum_i max_j (q_i . d_j), the TRF/ColBERT MaxSim operator."""
    sims = q_emb @ d_emb.T  # [n_query_tokens, n_doc_tokens]
    return sims.max(dim=1).values.sum().item()


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == False].reset_index(drop=True)
    print(f"Open-ended queries: {len(df)}", flush=True)

    retriever = hr.HybridRetriever()
    model = SentenceTransformer(ce.DEFAULT_MODEL)

    rows = []
    for i, row in df.iterrows():
        query = row["query"]
        bm25_cand = retriever._bm25_candidates(query)
        exact_match_ids = retriever._exact_match_ids(query)
        query_embedding = retriever.embedding_function.embed_query(query)
        vec_cand = retriever._vector_candidates(query, query_embedding=query_embedding)

        pool = retriever._score_linear(bm25_cand, vec_cand, exact_match_ids, 0.5)
        pool.sort(key=lambda x: x["score"], reverse=True)
        pool = pool[:TOP_K]

        q_tok = token_embeddings(model, query)
        trf_scored = []
        for c in pool:
            d_tok = token_embeddings(model, c["text"][:512])  # cap doc length, matches pooling truncation elsewhere
            trf_scored.append({**c, "trf_score": maxsim(q_tok, d_tok)})
        trf_scored.sort(key=lambda x: x["trf_score"], reverse=True)

        for label, results in [
            ("full_hybrid_baseline", pool),
            ("full_hybrid_trf_rerank", trf_scored),
        ]:
            rel_ranks = [j for j, c in enumerate(results) if is_relevant(c, query, str(row["reference_answer"]), False)]
            rows.append({
                "query_id": row["query_id"], "config": label,
                "recall@1": recall_at_k(rel_ranks, 1), "recall@3": recall_at_k(rel_ranks, 3),
                "recall@5": recall_at_k(rel_ranks, 5), "mrr": rr(rel_ranks),
                "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
            })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print("\n=== Recall/nDCG by config (open-ended, n=100) ===")
    print(out.groupby("config")[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
