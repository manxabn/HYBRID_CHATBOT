"""
Novel query-shape test, suggested directly by the user: compound queries
that require TWO separate structured facts from TWO different source
tables in the same answer (e.g. "What is the prerequisite for CSE221 and
who is the theory coordinator?" -- needs both a Prerequisites-table chunk
and a Coordinator-table chunk for the same course). This query shape has
never been evaluated in this project before; every existing entity-heavy
test query needs only one fact from one table.

Why this might behave differently from the existing entity-heavy result:
existing entity-heavy queries are solved almost entirely by BM25 exact-
match on a single course code, which is why full_hybrid ties bm25_only
there (see results/mcnemar_full_hybrid_vs_bm25.csv, 0 discordant pairs).
A compound query has two different targets that may not share strong
lexical overlap with the query text (the Coordinator chunk doesn't
literally contain the word "prerequisite", and vice versa) -- there IS a
real mechanism by which combining lexical and semantic signals could help
or hurt differently here than on single-fact queries. Reported honestly
either way; not run with an expectation of which config should win.

Relevance is binary per fact: does the top-k contain a chunk from table=
Prerequisites for this course, and (separately) a chunk from table=
Coordinator for this course. A query only "fully hits" if BOTH are present
in the same top-k -- the realistic bar for actually answering a compound
question, not just half of it.

Retrieval-only (no Ollama, no generation) -- cheap, safe to run any time.

Usage: python scripts/test_compound_queries.py
"""

import sys
from pathlib import Path

import pandas as pd
import sqlite3
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr

DB_PATH = ROOT / "knowledge_base.db"
OUT_RAW = ROOT / "results" / "compound_query_raw.csv"
OUT_MCNEMAR = ROOT / "results" / "compound_query_mcnemar.csv"
TOP_KS = [3, 5, 10]


def build_queries():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT p.Course, p.PreRequisite, c.FirstTheoryCoordinator, c.TheoryEmail
        FROM Prerequisites p JOIN Coordinator c ON p.Course = c.Course
        WHERE c.FirstTheoryCoordinator IS NOT NULL
        ORDER BY p.Course
    """)
    rows = cur.fetchall()
    conn.close()
    queries = []
    for course, prereq, coord, email in rows:
        queries.append({
            "query_id": f"COMPOUND-{course}",
            "query": f"What is the prerequisite for {course} and who is the theory coordinator?",
            "course": course,
            "expected_prereq": prereq,
            "expected_coordinator": coord,
        })
    return queries


def hit(results, k, course):
    top = results[:k]
    has_prereq = any(c["metadata"].get("table") == "Prerequisites" and c["metadata"].get("Course") == course for c in top)
    has_coord = any(c["metadata"].get("table") == "Coordinator" and c["metadata"].get("Course") == course for c in top)
    return has_prereq, has_coord, has_prereq and has_coord


def run_config(retriever, queries, config_fn, label):
    rows = []
    for q in queries:
        results = config_fn(retriever, q["query"])
        row = {"query_id": q["query_id"], "config": label}
        for k in TOP_KS:
            has_prereq, has_coord, both = hit(results, k, q["course"])
            row[f"prereq_hit@{k}"] = has_prereq
            row[f"coord_hit@{k}"] = has_coord
            row[f"both_hit@{k}"] = both
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar(a, b):
    a_wins = int(((a == 1) & (b == 0)).sum())
    b_wins = int(((a == 0) & (b == 1)).sum())
    n = a_wins + b_wins
    if n == 0:
        return a_wins, b_wins, n, 1.0
    p = binomtest(min(a_wins, b_wins), n, 0.5, alternative="two-sided").pvalue
    return a_wins, b_wins, n, p


def main():
    queries = build_queries()
    print(f"Built {len(queries)} compound (prerequisite + coordinator) queries", flush=True)

    retriever = hr.HybridRetriever()
    configs = {
        "bm25_only": lambda r, q: r.retrieve(q, 1.0, fusion="linear", top_n=10),
        "vector_only": lambda r, q: r.retrieve(q, 0.0, fusion="linear", top_n=10),
        "full_hybrid": lambda r, q: r.retrieve(q, 0.5, fusion="linear", top_n=10),
        "adaptive": lambda r, q: r.retrieve_adaptive(q, top_n=10)[0],
    }

    all_results = {}
    for label, fn in configs.items():
        print(f"Running {label}...", flush=True)
        all_results[label] = run_config(retriever, queries, fn, label)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(OUT_RAW, index=False)

    print("\n=== both_hit@k (needs BOTH prerequisite AND coordinator chunk in top-k) ===")
    summary = combined.groupby("config")[[f"both_hit@{k}" for k in TOP_KS]].mean().round(4)
    print(summary)
    print("\n=== prereq_hit@k and coord_hit@k separately ===")
    print(combined.groupby("config")[[f"prereq_hit@{k}" for k in TOP_KS] + [f"coord_hit@{k}" for k in TOP_KS]].mean().round(4))

    fh = all_results["full_hybrid"].set_index("query_id")
    bm = all_results["bm25_only"].set_index("query_id")
    vec = all_results["vector_only"].set_index("query_id")
    common = fh.index.intersection(bm.index)

    rows = []
    for other_label, other in [("bm25_only", bm), ("vector_only", vec)]:
        for k in TOP_KS:
            metric = f"both_hit@{k}"
            a = fh.loc[common, metric].astype(int).values
            b = other.loc[common, metric].astype(int).values
            fh_wins, other_wins, n_disc, p = mcnemar(a, b)
            rows.append({
                "comparison": f"full_hybrid_vs_{other_label}", "metric": metric, "n": len(common),
                "full_hybrid_only_correct": fh_wins, f"{other_label}_only_correct": other_wins,
                "n_discordant": n_disc, "mcnemar_p": round(p, 4), "significant_at_0.05": p < 0.05,
            })
            print(f"[full_hybrid vs {other_label}] {metric}: full_hybrid-only={fh_wins}, "
                  f"{other_label}-only={other_wins}, n_discordant={n_disc}, p={p:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MCNEMAR, index=False)
    print(f"\nWrote {OUT_RAW} and {OUT_MCNEMAR}")


if __name__ == "__main__":
    main()
