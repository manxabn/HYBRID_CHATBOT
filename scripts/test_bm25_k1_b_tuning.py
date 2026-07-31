"""
Item #49 from a literature-derived checklist: confirm the BM25 baseline
uses tuned k1/b, not silently-accepted library defaults. This project's
`scripts/build_bm25_index.py` calls `BM25Okapi(tokenized)` with no k1/b
arguments -- `rank_bm25`'s defaults (k1=1.5, b=0.75), never checked
against this specific corpus's chunk-length distribution (short, fairly
uniform, 500-char-window structured/QA chunks, not general web documents
BM25's classic defaults were tuned on).

Sweeps a small k1/b grid, scored on pure BM25 ranking quality (recall@1/
5, no exact-match ceiling involved, to isolate BM25's own lexical scoring
specifically) against the same 200-query test set and relevance judgment
already used in scripts/measure_ir_metrics.py.

Retrieval-only, no Ollama, no GPU needed for BM25 itself.

Usage: python scripts/test_bm25_k1_b_tuning.py
"""

import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.measure_ir_metrics import is_relevant, recall_at_k, rr

BM25_PKL = ROOT / "data" / "bm25_corpus.pkl"
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "bm25_k1_b_sweep.csv"
TOP_K = 10

TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def main():
    with open(BM25_PKL, "rb") as f:
        data = pickle.load(f)
    doc_ids = data["doc_ids"]
    texts = data["texts"]
    tokenized = [tokenize(t) for t in texts]

    corpus_by_id = {}
    with open(ROOT / "data" / "corpus.jsonl", encoding="utf-8") as f:
        import json
        for line in f:
            rec = json.loads(line)
            corpus_by_id[rec["doc_id"]] = rec

    df = pd.read_csv(QUERIES_PATH)

    grid = [(1.5, 0.75), (1.2, 0.75), (0.9, 0.4), (1.5, 0.4), (0.5, 0.3), (2.0, 0.9)]
    rows = []
    for k1, b in grid:
        bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        n_r1, n_r5, n = 0, 0, 0
        n_r1_ent, n_r5_ent, n_ent = 0, 0, 0
        n_r1_open, n_r5_open, n_open = 0, 0, 0
        for _, row in df.iterrows():
            q_tokens = tokenize(row["query"])
            scores = bm25.get_scores(q_tokens)
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
            candidates = []
            for idx in ranked_idx:
                doc_id = doc_ids[idx]
                rec = corpus_by_id.get(doc_id, {"text": "", "metadata": {}})
                candidates.append(rec)
            rel_ranks = [i for i, c in enumerate(candidates)
                         if is_relevant(c, row["query"], str(row["reference_answer"]), row["is_entity_heavy"])]
            r1, r5 = recall_at_k(rel_ranks, 1), recall_at_k(rel_ranks, 5)
            n_r1 += r1; n_r5 += r5; n += 1
            if row["is_entity_heavy"]:
                n_r1_ent += r1; n_r5_ent += r5; n_ent += 1
            else:
                n_r1_open += r1; n_r5_open += r5; n_open += 1
        row_result = {
            "k1": k1, "b": b, "n": n,
            "recall@1_all": round(n_r1 / n, 4), "recall@5_all": round(n_r5 / n, 4),
            "recall@1_entity_heavy": round(n_r1_ent / n_ent, 4), "recall@5_entity_heavy": round(n_r5_ent / n_ent, 4),
            "recall@1_open_ended": round(n_r1_open / n_open, 4), "recall@5_open_ended": round(n_r5_open / n_open, 4),
        }
        rows.append(row_result)
        print(f"k1={k1} b={b}: recall@1={row_result['recall@1_all']} recall@5={row_result['recall@5_all']} "
              f"(entity r@1={row_result['recall@1_entity_heavy']}, open r@1={row_result['recall@1_open_ended']})",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print("\nNote: this is PURE BM25 ranking (no exact-match ceiling), to isolate lexical-scoring quality itself.")


if __name__ == "__main__":
    main()
