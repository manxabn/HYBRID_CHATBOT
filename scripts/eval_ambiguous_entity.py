"""
Retrieval-only evaluation of the ambiguous/newly-reachable partial-name
faculty queries (data/test_queries_ambiguous_entity.csv, built by
scripts/build_ambiguous_entity_test.py) across bm25_only, full_hybrid, and
adaptive fusion -- the first evaluation on this project's data where
fusion-method choice has anything real to decide, since every query in the
main 200-query test set resolves to exactly one exact match (Sec.
adaptive-isolated / rrf-k-sweep both found 100/100 ties for exactly this
reason).

Two different questions, asked separately since they have different notions
of "correct":
  - Ambiguous queries (n_true_candidates > 1): there is no single right
    top-1 answer by construction. We measure SET RECALL -- what fraction of
    the true candidate set appears in the top-k -- since a good system
    should surface all plausible candidates for a genuinely ambiguous
    query, not silently commit to one.
  - Unambiguous-but-partial-name queries (n_true_candidates == 1): a
    standard Recall@1/MRR question, same as any other unambiguous-match
    test, just newly reachable because of the token-index fix.

Usage: python scripts/eval_ambiguous_entity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr

QUERIES_PATH = ROOT / "data" / "test_queries_ambiguous_entity.csv"
OUT_PATH = ROOT / "results" / "ambiguous_entity_eval.csv"
TOP_K = 10


def rank_of_true_ids(results, true_ids):
    ranked_doc_ids = [c["doc_id"] for c in results[:TOP_K]]
    return [ranked_doc_ids.index(t) if t in ranked_doc_ids else None for t in true_ids]


def main():
    df = pd.read_csv(QUERIES_PATH)
    retriever = hr.HybridRetriever()

    configs = {
        "bm25_only": lambda q: retriever.retrieve(q, 1.0, fusion="linear", top_n=TOP_K),
        "full_hybrid": lambda q: retriever.retrieve(q, 0.5, fusion="linear", top_n=TOP_K),
        "adaptive_rrf": lambda q: retriever.retrieve(q, 0.9, fusion="rrf", top_n=TOP_K),
    }

    rows = []
    for _, r in df.iterrows():
        true_ids = str(r["true_doc_ids"]).split("|")
        for label, fn in configs.items():
            results = fn(r["query"])
            ranks = rank_of_true_ids(results, true_ids)
            n_found = sum(1 for rk in ranks if rk is not None)
            top1_doc_id = results[0]["doc_id"] if results else None
            rows.append({
                "query_id": r["query_id"], "config": label, "query": r["query"],
                "is_ambiguous": r["is_ambiguous"], "n_true_candidates": r["n_true_candidates"],
                "n_found_in_topk": n_found, "set_recall": n_found / len(true_ids),
                "top1_doc_id": top1_doc_id,
                "top1_is_true_candidate": top1_doc_id in true_ids,
            })
        print(f"  {r['query_id']} ({r['n_true_candidates']} true) done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("\n=== Unambiguous partial-name queries (n_true_candidates == 1): Recall@1-equivalent ===")
    unambig = out[~out["is_ambiguous"]]
    print(unambig.groupby("config")["top1_is_true_candidate"].mean().round(4))

    print("\n=== Ambiguous queries (n_true_candidates > 1): mean SET recall@10 ===")
    ambig = out[out["is_ambiguous"]]
    print(ambig.groupby("config")["set_recall"].mean().round(4))

    print("\n=== Ambiguous queries: does fusion method change WHICH candidate ranks top-1? ===")
    top1_by_config = out[out["is_ambiguous"]].pivot(index="query_id", columns="config", values="top1_doc_id")
    print(f"bm25_only vs full_hybrid: {(top1_by_config['bm25_only'] != top1_by_config['full_hybrid']).sum()}/{len(top1_by_config)} queries differ")
    print(f"bm25_only vs adaptive_rrf: {(top1_by_config['bm25_only'] != top1_by_config['adaptive_rrf']).sum()}/{len(top1_by_config)} queries differ")
    print(f"full_hybrid vs adaptive_rrf: {(top1_by_config['full_hybrid'] != top1_by_config['adaptive_rrf']).sum()}/{len(top1_by_config)} queries differ")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
