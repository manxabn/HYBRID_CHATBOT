"""
Tests whether a semantic (vector-similarity) sufficiency signal can rescue
the paraphrase-robustness failure found in results/paraphrase_robustness_
raw.csv (20/20 open-ended paraphrases incorrectly abstained) WITHOUT
falsely rescuing genuinely out-of-scope queries -- the real risk of any
quick fix to the abstention gate. Uses the same out-of-scope source data
as scripts/calibrate_abstention.py (EnglishQA/BanglishQA Category='Out of
Scope / Unanswerable'), not invented negatives.

For each query, computes max(s_vec) across the retrieved candidate pool --
the vector-similarity component alone, independent of query_top1_score's
BM25-heavy composite that was confirmed to be what drops under
paraphrasing. Reports the s_vec distribution for (a) the 20 abstained
paraphrases that SHOULD be rescued and (b) a real out-of-scope sample
that should NOT be rescued, so a threshold choice is evidence-based, not
guessed.

Usage: python scripts/test_semantic_sufficiency_override.py
"""

import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr

DB_PATH = ROOT / "knowledge_base.db"
PARAPHRASE_RAW = ROOT / "results" / "paraphrase_robustness_raw.csv"
OUT_PATH = ROOT / "results" / "semantic_sufficiency_check.csv"
SAMPLE_SEED = 42
N_OOS_SAMPLE = 40


def load_out_of_scope_sample(n):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT Question, Category FROM EnglishQA")
    rows = cur.fetchall()
    conn.close()
    oos = [q for q, cat in rows if cat == "Out of Scope / Unanswerable" and q]
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(oos, min(n, len(oos)))


def max_s_vec(retriever, query, top_n=10):
    results = retriever.retrieve(query, 0.5, fusion="linear", top_n=top_n)
    if not results:
        return 0.0
    return max(c["s_vec"] for c in results)


def main():
    retriever = hr.HybridRetriever()

    para = pd.read_csv(PARAPHRASE_RAW)
    abstained_paraphrases = para[(para["abstained"] == True) & (~para["query_id"].str.endswith("-orig"))]
    print(f"Abstained paraphrases to check: {len(abstained_paraphrases)}")

    rows = []
    for _, r in abstained_paraphrases.iterrows():
        s = max_s_vec(retriever, r["query"])
        rows.append({"group": "abstained_paraphrase_SHOULD_rescue", "query_id": r["query_id"],
                      "query": r["query"], "max_s_vec": s})

    oos_queries = load_out_of_scope_sample(N_OOS_SAMPLE)
    print(f"Out-of-scope queries to check: {len(oos_queries)}")
    for i, q in enumerate(oos_queries):
        s = max_s_vec(retriever, q)
        rows.append({"group": "out_of_scope_SHOULD_NOT_rescue", "query_id": f"OOS-{i}",
                      "query": q, "max_s_vec": s})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("\n=== max_s_vec distribution by group ===")
    print(out.groupby("group")["max_s_vec"].describe())

    print("\n=== candidate thresholds ===")
    for t in [0.55, 0.60, 0.65, 0.70, 0.75]:
        resc = (out[out["group"].str.contains("SHOULD_rescue")]["max_s_vec"] >= t).mean()
        false_resc = (out[out["group"].str.contains("SHOULD_NOT_rescue")]["max_s_vec"] >= t).mean()
        print(f"threshold={t}: rescues {resc*100:.0f}% of paraphrases, "
              f"falsely rescues {false_resc*100:.0f}% of out-of-scope queries")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
