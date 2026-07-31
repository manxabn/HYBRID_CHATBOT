"""
Uses a real, unused piece of this corpus's own structure: every EnglishQA
row has a Category label (Admission, Campus, Library, ...), stored in
corpus metadata but never touched by retrieval scoring anywhere in
pipeline/hybrid_retriever.py -- BM25 and vector search both operate on
chunk TEXT only. This tests whether a category-match signal, computed
from the SAME embedding model already in use (no new training), adds
real value on top of BM25+vector fusion.

Method: build one "category centroid" embedding per category by averaging
the embeddings of that category's own TRAIN-split questions (never test
-split, so this doesn't leak test information). For each test query,
find its nearest category centroid (a free, zero-shot classification
using the existing embedding space), then boost candidates whose own
Category metadata matches the predicted category.

Open-ended queries only (n=100), same reasoning as HyDE/grounded-PRF:
entity-heavy is saturated by exact-match regardless of any other signal.
Retrieval-only, no Ollama needed.

Usage: python scripts/test_category_boost.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from scripts.measure_ir_metrics import is_relevant, recall_at_k, rr, ndcg_at_k, TOP_K

DB_PATH = ROOT / "knowledge_base.db"
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "category_boost_raw.csv"
CATEGORY_BOOST = 0.15  # additive boost, same order of magnitude as EXACT_MATCH_BONUS (0.3)


def build_category_centroids(retriever):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT Category, Question FROM EnglishQA WHERE Split='train' AND Category != 'Out of Scope / Unanswerable'")
    rows = cur.fetchall()
    conn.close()

    by_cat = {}
    for cat, q in rows:
        by_cat.setdefault(cat, []).append(q)

    centroids = {}
    for cat, questions in by_cat.items():
        sample = questions[:40]  # cap for speed, still a real, representative sample
        embs = [np.array(retriever.embedding_function.embed_query(q)) for q in sample]
        centroid = np.mean(embs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[cat] = centroid
    return centroids


def predict_category(query_embedding, centroids):
    best_cat, best_sim = None, -2.0
    qe = np.array(query_embedding)
    qe = qe / (np.linalg.norm(qe) + 1e-8)
    for cat, centroid in centroids.items():
        sim = float(np.dot(qe, centroid))
        if sim > best_sim:
            best_sim, best_cat = sim, cat
    return best_cat, best_sim


def main():
    retriever = hr.HybridRetriever()
    print("Building category centroids from train-split questions...", flush=True)
    centroids = build_category_centroids(retriever)
    print(f"Built {len(centroids)} category centroids", flush=True)

    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == False].reset_index(drop=True)
    print(f"Open-ended queries: {len(df)}", flush=True)

    rows = []
    correct_predictions = 0
    for _, row in df.iterrows():
        query = row["query"]
        query_embedding = retriever.embedding_function.embed_query(query)
        predicted_cat, sim = predict_category(query_embedding, centroids)

        bm25_cand = retriever._bm25_candidates(query)
        exact_match_ids = retriever._exact_match_ids(query)
        vec_cand = retriever._vector_candidates(query, query_embedding=query_embedding)

        for label, lam, use_boost in [
            ("bm25_only", 1.0, False),
            ("full_hybrid_baseline", 0.5, False),
            ("full_hybrid_category_boost", 0.5, True),
            ("vector_only_baseline", 0.0, False),
            ("vector_only_category_boost", 0.0, True),
        ]:
            scored = retriever._score_linear(bm25_cand, vec_cand, exact_match_ids, lam)
            if use_boost:
                for c in scored:
                    if c["metadata"].get("table") == "EnglishQA" and c["metadata"].get("Category") == predicted_cat:
                        c["score"] += CATEGORY_BOOST
            scored.sort(key=lambda x: x["score"], reverse=True)
            results = scored[:TOP_K]
            rel_ranks = [i for i, c in enumerate(results) if is_relevant(c, query, str(row["reference_answer"]), False)]
            rows.append({
                "query_id": row["query_id"], "config": label, "predicted_category": predicted_cat,
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
