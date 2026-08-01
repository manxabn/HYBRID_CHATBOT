"""
Follow-up to scripts/test_trf_maxsim_rerank.py: that experiment found a
real, significant retrieval-level nDCG improvement from MaxSim re-ranking
(p=0.006-0.013) but recall@1/MRR barely moved (2/100 queries changed) --
this project's own BM25-tuning experience earlier this session showed
ranking-order changes often do NOT move downstream generation quality
once the right chunk is present either way, so that result was explicitly
flagged as "not yet known" rather than assumed to transfer. This script
tests it directly rather than leave it assumed.

Method: for the same 100 open-ended queries, build context from the top
final_k=5 candidates under two orderings -- the baseline full_hybrid
ranking, and the MaxSim-reordered ranking (same pool of 10, same
underlying candidates, only the ORDER within the top-5 can differ, plus
which candidates make the top-5 cut if MaxSim promoted something from
rank 6-10) -- generate a real answer with the deployed LLM for each, and
score BLEU/ROUGE-L/BERTScore/METEOR.

Usage: python scripts/test_trf_generation_quality.py
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
import pipeline.chroma_embedding as ce
from pipeline.ollama_client import generate
from scripts.test_trf_maxsim_rerank import token_embeddings, maxsim

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "trf_generation_quality_raw.csv"
FINAL_K = 5
FIELDNAMES = ["query_id", "config", "query", "reference_answer", "generated_answer"]


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == False].reset_index(drop=True)
    print(f"Open-ended queries: {len(df)}", flush=True)

    completed = set()
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        prior = pd.read_csv(OUT_PATH)
        completed = set(zip(prior["query_id"], prior["config"]))
        print(f"Resuming: {len(completed)} (query_id, config) pairs already done", flush=True)
    write_header = (not OUT_PATH.exists()) or OUT_PATH.stat().st_size == 0

    retriever = hr.HybridRetriever()
    model = SentenceTransformer(ce.DEFAULT_MODEL)

    with open(OUT_PATH, "a" if not write_header else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, row in df.iterrows():
            query = row["query"]
            ref = row["reference_answer"]

            bm25_cand = retriever._bm25_candidates(query)
            exact_match_ids = retriever._exact_match_ids(query)
            query_embedding = retriever.embedding_function.embed_query(query)
            vec_cand = retriever._vector_candidates(query, query_embedding=query_embedding)

            pool = retriever._score_linear(bm25_cand, vec_cand, exact_match_ids, 0.5)
            pool.sort(key=lambda x: x["score"], reverse=True)
            pool = pool[:10]

            q_tok = token_embeddings(model, query)
            trf_scored = []
            for c in pool:
                d_tok = token_embeddings(model, c["text"][:512])
                trf_scored.append({**c, "trf_score": maxsim(q_tok, d_tok)})
            trf_scored.sort(key=lambda x: x["trf_score"], reverse=True)

            for label, results in [
                ("full_hybrid_baseline", pool),
                ("full_hybrid_trf_rerank", trf_scored),
            ]:
                if (row["query_id"], label) in completed:
                    continue
                context = "\n\n".join(d["text"] for d in results[:FINAL_K])
                answer = generate(query, context)
                writer.writerow({
                    "query_id": row["query_id"], "config": label,
                    "query": query, "reference_answer": ref, "generated_answer": answer,
                })
                f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(df)} done", flush=True)

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
