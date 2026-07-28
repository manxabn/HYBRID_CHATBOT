"""
Cross-encoder reranker for the fused BM25+vector candidate set.

Rationale: hybrid_retriever.py's fusion (linear or RRF) scores each stream
independently and combines the two scores -- it never looks at query and
document text *together*. A cross-encoder does exactly that: it scores each
(query, candidate) pair jointly, which is a strictly more expressive
relevance signal than combining two independently-computed scores, at the
cost of one forward pass per candidate. Well-evidenced in the reranking
literature as a cheap, high-value addition to a two-stream retriever (see
project literature review). We use a small (33M-param) CPU-friendly
cross-encoder rather than a larger one specifically because the added
latency must stay negligible against this project's already-reported
end-to-end latency (paper.tex Section 4.6: 1.80s mean, generation-dominated)
-- reranking a top-10 candidate set costs single-digit milliseconds per
candidate, not seconds.
"""

from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Domain-fine-tuned reranker (scripts/finetune_reranker.py, 2026-07-28):
# trained on this corpus's own hard negatives (real retrieved near-misses,
# not random text) to directly address the confirmed regression the generic
# model caused once embeddings were fine-tuned (results/significance_tests_
# novel_roundK.csv: reranker-on lost significantly to full_hybrid on
# BLEU/METEOR; reranker-off did not). Same auto-detect-local-else-fall-back
# pattern as chroma_embedding.py's DEFAULT_MODEL.
_FINETUNED_RERANKER_PATH = Path(__file__).resolve().parent.parent / "models" / "finetuned_reranker_domain"
DEFAULT_RERANKER_MODEL = str(_FINETUNED_RERANKER_PATH) if _FINETUNED_RERANKER_PATH.exists() else RERANKER_MODEL


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        # Explicit device selection (see embeddings.py) -- this model is
        # tiny (~90MB) and easily fits in VRAM alongside Ollama's resident
        # model, rather than relying on implicit auto-detection.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """`candidates` is a list of the dicts hybrid_retriever.py's
        _score_linear/_score_rrf produce (each has a "text" field). Returns
        the top_k candidates re-sorted by cross-encoder score, with a new
        "rerank_score" field added to every input candidate (not just the
        survivors) so the full pre-rerank ranking stays inspectable."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
