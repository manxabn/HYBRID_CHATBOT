"""
Expanded compound-query test, adding real statistical power to the
promising-but-underpowered n=26 result in scripts/test_compound_queries.py
(both_hit@3: full_hybrid 1.000 vs bm25_only 0.885, clean 3-0 sweep, but
p=0.25 -- not enough discordant pairs to confirm).

Adds a SECOND, much larger compound-query type: "who teaches the theory
section of {course} and what is their office room?" -- requires a
CourseDetails chunk (course -> TheoryInitial) AND a FacultyList chunk
(that Initial -> Room) in the same top-k. Built from every CourseDetails
row whose TheoryInitial resolves to a real FacultyList entry (541 real
joined rows, no invented facts, no subsampling to cherry-pick a favorable
slice -- the full available set is used).

Combined with the original 26 (course, prerequisite+coordinator) queries,
this gives n=567 total compound queries across two genuinely different
table-pair combinations, both requiring two separate structured facts
that don't share much lexical overlap with each other.

Retrieval-only (no Ollama, no generation).

Usage: python scripts/test_compound_queries_expanded.py
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
OUT_RAW = ROOT / "results" / "compound_query_expanded_raw.csv"
OUT_MCNEMAR = ROOT / "results" / "compound_query_expanded_mcnemar.csv"
TOP_KS = [3, 5, 10]


def build_queries():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    queries = []

    # Type A: prerequisite + theory coordinator (same as test_compound_queries.py)
    cur.execute("""
        SELECT p.Course, p.PreRequisite, c.FirstTheoryCoordinator
        FROM Prerequisites p JOIN Coordinator c ON p.Course = c.Course
        WHERE c.FirstTheoryCoordinator IS NOT NULL
        ORDER BY p.Course
    """)
    for course, prereq, coord in cur.fetchall():
        queries.append({
            "query_id": f"COMPOUND-A-{course}",
            "type": "prereq_coordinator",
            "query": f"What is the prerequisite for {course} and who is the theory coordinator?",
            "check": ("Prerequisites", course, "Coordinator", course),
        })

    # Type B: theory instructor + their office room (course -> faculty -> room)
    cur.execute("""
        SELECT cd.Course, cd.TheoryInitial, fl.Name, fl.Room
        FROM CourseDetails cd JOIN FacultyList fl ON cd.TheoryInitial = fl.Initial
        WHERE cd.TheoryInitial IS NOT NULL AND cd.TheoryInitial != ''
        ORDER BY cd.Course
    """)
    for course, initial, name, room in cur.fetchall():
        queries.append({
            "query_id": f"COMPOUND-B-{course}",
            "type": "faculty_room",
            "query": f"Who teaches the theory section of {course} and what is their office room?",
            "check": ("CourseDetails", course, "FacultyList", initial),
        })

    conn.close()
    return queries


def hit(results, k, table1, key1, table2, key2):
    top = results[:k]
    has1 = any(c["metadata"].get("table") == table1 and
               (c["metadata"].get("Course") == key1 if table1 != "FacultyList" else c["metadata"].get("Initial") == key1)
               for c in top)
    has2 = any(c["metadata"].get("table") == table2 and
               (c["metadata"].get("Initial") == key2 if table2 == "FacultyList" else c["metadata"].get("Course") == key2)
               for c in top)
    return has1 and has2


def run_config(retriever, queries, config_fn, label):
    rows = []
    for q in queries:
        results = config_fn(retriever, q["query"])
        t1, k1, t2, k2 = q["check"]
        row = {"query_id": q["query_id"], "config": label, "type": q["type"]}
        for k in TOP_KS:
            row[f"both_hit@{k}"] = hit(results, k, t1, k1, t2, k2)
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
    print(f"Built {len(queries)} compound queries "
          f"({sum(1 for q in queries if q['type']=='prereq_coordinator')} prereq+coordinator, "
          f"{sum(1 for q in queries if q['type']=='faculty_room')} faculty+room)", flush=True)

    retriever = hr.HybridRetriever()
    configs = {
        "bm25_only": lambda r, q: r.retrieve(q, 1.0, fusion="linear", top_n=10),
        "full_hybrid": lambda r, q: r.retrieve(q, 0.5, fusion="linear", top_n=10),
    }

    all_results = {}
    for label, fn in configs.items():
        print(f"Running {label} on {len(queries)} queries...", flush=True)
        all_results[label] = run_config(retriever, queries, fn, label)
        print(f"  done", flush=True)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(OUT_RAW, index=False)

    print("\n=== both_hit@k by type ===")
    print(combined.groupby(["config", "type"])[[f"both_hit@{k}" for k in TOP_KS]].mean().round(4))
    print("\n=== both_hit@k, all types combined ===")
    print(combined.groupby("config")[[f"both_hit@{k}" for k in TOP_KS]].mean().round(4))

    fh = all_results["full_hybrid"].set_index("query_id")
    bm = all_results["bm25_only"].set_index("query_id")
    common = fh.index.intersection(bm.index)

    rows = []
    for subset_name, subset_ids in [
        ("all", common),
        ("prereq_coordinator", [i for i in common if fh.loc[i, "type"] == "prereq_coordinator"]),
        ("faculty_room", [i for i in common if fh.loc[i, "type"] == "faculty_room"]),
    ]:
        for k in TOP_KS:
            metric = f"both_hit@{k}"
            a = fh.loc[subset_ids, metric].astype(int).values
            b = bm.loc[subset_ids, metric].astype(int).values
            fh_wins, bm_wins, n_disc, p = mcnemar(a, b)
            rows.append({
                "subset": subset_name, "metric": metric, "n": len(subset_ids),
                "full_hybrid_only_correct": fh_wins, "bm25_only_only_correct": bm_wins,
                "n_discordant": n_disc, "mcnemar_p": round(p, 4), "significant_at_0.05": p < 0.05,
            })
            print(f"[{subset_name}] {metric}: n={len(subset_ids)} full_hybrid-only={fh_wins}, "
                  f"bm25_only-only={bm_wins}, n_discordant={n_disc}, p={p:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MCNEMAR, index=False)
    print(f"\nWrote {OUT_RAW} and {OUT_MCNEMAR}")


if __name__ == "__main__":
    main()
