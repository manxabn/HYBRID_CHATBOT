"""
Build data/test_queries_banglish.csv: a BanglishQA-sourced test set, same
methodology as build_test_queries.py's sample_open_ended() but for the
Banglish table -- this pipeline has never been evaluated on Banglish/code-
mixed queries at all (every round A-J used English-only test queries).

BanglishQA has no Type (Original/Paraphrase) column like EnglishQA, but the
same (Category, Answer) exact-grouping is applied for consistency/rigor
regardless -- confirmed empty-handed here (111 distinct groups across 111
eligible rows, i.e. no duplication at all in this table's test split),
matching the earlier finding that BanglishQA is far less paraphrase-
duplicated than EnglishQA (~9% vs ~82%).

Questions are Banglish (code-mixed Bengali/English in Latin script);
reference answers are in English (AnswerEnglish column) -- that's the
dataset's own design, not a script choice.

Usage: python scripts/build_banglish_test_queries.py
"""

import random
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "data" / "test_queries_banglish.csv"
SEED = 42
N_QUERIES = 100


def main():
    rng = random.Random(SEED)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT id, Category, QuestionBanglish, AnswerEnglish FROM BanglishQA "
        "WHERE QuestionBanglish IS NOT NULL AND AnswerEnglish IS NOT NULL "
        "AND Split = 'test' AND Category != 'Out of Scope / Unanswerable'",
        conn,
    )
    conn.close()
    df = df[(df["QuestionBanglish"].str.strip() != "") & (df["AnswerEnglish"].str.strip() != "")]
    df["cluster_id"] = df["Category"].str.strip() + "||" + df["AnswerEnglish"].str.strip()

    cluster_ids = df["cluster_id"].unique().tolist()
    chosen = rng.sample(cluster_ids, min(N_QUERIES, len(cluster_ids)))
    rows = []
    for cid in chosen:
        candidates = df[df["cluster_id"] == cid]
        r = candidates.iloc[rng.randrange(len(candidates))]
        rows.append({
            "query": r["QuestionBanglish"].strip(),
            "reference_answer": r["AnswerEnglish"].strip(),
            "is_entity_heavy": False,
            "source": "BanglishQA",
        })

    out = pd.DataFrame(rows)
    out.insert(0, "query_id", [f"BQ{i+1:03d}" for i in range(len(out))])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} Banglish test queries to {OUT_PATH}")


if __name__ == "__main__":
    main()
