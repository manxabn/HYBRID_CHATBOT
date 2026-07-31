"""
Third compound-query type: prerequisite + theory room, for the same
course section (Prerequisites.Course + CourseDetails.TheoryRoom, real
join, 416 available). Unlike the faculty+room type (0/540 for both
configs -- a structural retrieval-scope gap, not a fusion question,
already fixed separately via pipeline/faculty_room_lookup.py), both facts
here are plain single-hop retrieval targets: a Prerequisites chunk and a
CourseDetails chunk, each independently findable by a flat retrieval pass
with no second lookup required. Checking whether THIS type is actually
retrievable by both configs before committing to a full run -- worth
running only if it isn't another 0%-for-both structural gap.

Usage: python scripts/test_compound_prereq_room.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
import sqlite3
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr

DB_PATH = ROOT / "knowledge_base.db"
OUT_RAW = ROOT / "results" / "compound_prereq_room_raw.csv"
OUT_MCNEMAR = ROOT / "results" / "compound_prereq_room_mcnemar.csv"
TOP_KS = [3, 5, 10]
SAMPLE_SEED = 42


def build_queries(n=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT p.Course, cd.Course, cd.TheoryRoom
        FROM Prerequisites p JOIN CourseDetails cd ON cd.Course LIKE p.Course || '-%'
        WHERE cd.TheoryRoom IS NOT NULL AND cd.TheoryRoom != ''
        ORDER BY p.Course, cd.Course
    """)
    rows = cur.fetchall()
    conn.close()
    if n is not None:
        rng = random.Random(SAMPLE_SEED)
        rows = rng.sample(rows, min(n, len(rows)))
    queries = []
    for base_course, section_course, room in rows:
        queries.append({
            "query_id": f"COMPOUND-C-{section_course}",
            "query": f"What is the prerequisite for {section_course} and which room is its theory class held in?",
            "check": ("Prerequisites", base_course, "CourseDetails", section_course),
        })
    return queries


def hit(results, k, table1, key1, table2, key2):
    top = results[:k]
    has1 = any(c["metadata"].get("table") == table1 and c["metadata"].get("Course") == key1 for c in top)
    has2 = any(c["metadata"].get("table") == table2 and c["metadata"].get("Course") == key2 for c in top)
    return has1 and has2


def run_config(retriever, queries, config_fn, label):
    rows = []
    for q in queries:
        results = config_fn(retriever, q["query"])
        t1, k1, t2, k2 = q["check"]
        row = {"query_id": q["query_id"], "config": label}
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
    print(f"Built {len(queries)} prereq+room compound queries", flush=True)

    retriever = hr.HybridRetriever()
    configs = {
        "bm25_only": lambda r, q: r.retrieve(q, 1.0, fusion="linear", top_n=10),
        "full_hybrid": lambda r, q: r.retrieve(q, 0.5, fusion="linear", top_n=10),
    }

    all_results = {}
    for label, fn in configs.items():
        print(f"Running {label} on {len(queries)} queries...", flush=True)
        all_results[label] = run_config(retriever, queries, fn, label)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(OUT_RAW, index=False)
    print("\n=== both_hit@k ===")
    print(combined.groupby("config")[[f"both_hit@{k}" for k in TOP_KS]].mean().round(4))

    fh = all_results["full_hybrid"].set_index("query_id")
    bm = all_results["bm25_only"].set_index("query_id")
    common = fh.index.intersection(bm.index)

    rows = []
    for k in TOP_KS:
        metric = f"both_hit@{k}"
        a = fh.loc[common, metric].astype(int).values
        b = bm.loc[common, metric].astype(int).values
        fh_wins, bm_wins, n_disc, p = mcnemar(a, b)
        rows.append({
            "metric": metric, "n": len(common),
            "full_hybrid_only_correct": fh_wins, "bm25_only_only_correct": bm_wins,
            "n_discordant": n_disc, "mcnemar_p": round(p, 4), "significant_at_0.05": p < 0.05,
        })
        print(f"[{metric}] n={len(common)} full_hybrid-only={fh_wins}, bm25_only-only={bm_wins}, "
              f"n_discordant={n_disc}, p={p:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MCNEMAR, index=False)
    print(f"\nWrote {OUT_RAW} and {OUT_MCNEMAR}")


if __name__ == "__main__":
    main()
