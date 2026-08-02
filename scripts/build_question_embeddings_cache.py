"""
Precomputes and caches the question-only embeddings HybridRetriever uses
for its question-field-aware similarity boost (see hybrid_retriever.py's
module docstring). Same pattern as scripts/build_bm25_index.py: build once
here, load from disk at __init__ time instead of re-embedding ~5,400
texts on every single process start (real, measured startup cost found
during a 2026-08-01 code audit -- every eval script, every server
restart, paid this again from scratch, unlike the main Chroma index and
the BM25 index, both of which were already persisted).

Cache invalidation: the saved file also stores a fingerprint (embedding
model name + a hash of the exact (doc_id, question_text) pairs embedded)
so HybridRetriever can detect a stale cache (corpus changed, or the
deployed embedding model changed) and fall back to live computation
--with a printed warning -- rather than silently serve stale vectors.

Usage: python scripts/build_question_embeddings_cache.py
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.chroma_embedding import Chroma1xEmbeddingFunction, DEFAULT_MODEL

CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
OUT_PATH = ROOT / "data" / "question_embeddings_cache.pkl"


def compute_fingerprint(model_name: str, doc_ids: list, texts: list) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for doc_id, text in zip(doc_ids, texts):
        h.update(doc_id.encode("utf-8"))
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def main():
    question_doc_ids, question_texts = [], []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec["metadata"]
            q_text = meta.get("Question") or meta.get("QuestionBanglish")
            if q_text:
                question_doc_ids.append(rec["doc_id"])
                question_texts.append(q_text)

    print(f"Embedding {len(question_texts)} question-only texts under {DEFAULT_MODEL}...", flush=True)
    embedding_function = Chroma1xEmbeddingFunction()
    question_vecs = embedding_function(question_texts)

    fingerprint = compute_fingerprint(DEFAULT_MODEL, question_doc_ids, question_texts)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "model_name": DEFAULT_MODEL,
            "fingerprint": fingerprint,
            "doc_ids": question_doc_ids,
            "vectors": question_vecs,
        }, f)

    print(f"Wrote {OUT_PATH} ({len(question_doc_ids)} vectors)")


if __name__ == "__main__":
    main()
