"""
Table-routing coverage audit -- the generalized fix for weakness #9: this
project found the exact same class of bug twice (Prerequisites/Coordinator
vs. CourseDetails; FacultyList vs. FacultyAvailability), in both cases
because a structured table existed in the corpus with NO query in any test
set that actually required and correctly routed to it -- the gap was only
found by accident (building a dedicated test set for the second one).

This script converts that from a reactive, accidental discovery pattern
into a standing, automatable check, in the spirit of code-coverage tooling
applied to table schema instead of code branches: for every table in the
corpus, does at least one query across all EXISTING test-query CSVs
actually retrieve a chunk FROM that table at rank 1 (i.e. is the table not
just present in the corpus, but demonstrably reachable and preferred by
the retriever for at least one real query)? A table with zero such queries
is a documented, flagged coverage gap -- exactly the shape of both prior
incidents -- rather than a silent assumption that routing "just works".

This is a coverage AUDIT, not a coverage GUARANTEE: passing does not prove
routing is correct for that table, only that it has been exercised at all.
Both prior bugs were in tables that had ZERO exercising queries; this
check would have flagged both before they were found by accident.

Usage: python scripts/audit_table_coverage.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever

QUERY_FILES = [
    ROOT / "data" / "test_queries.csv",
    ROOT / "data" / "test_queries_banglish.csv",
    ROOT / "data" / "test_queries_faculty.csv",
    ROOT / "data" / "test_queries_crosslingual_stress.csv",
]

ALL_CORPUS_TABLES = [
    "EnglishQA", "BanglishQA", "CourseDetails", "FacultyList",
    "FacultyAvailability", "Coordinator", "Prerequisites",
]

OUT_PATH = ROOT / "results" / "table_coverage_audit.csv"


def main():
    retriever = HybridRetriever()

    queries = []
    for path in QUERY_FILES:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        df = pd.read_csv(path)
        if "query" not in df.columns:
            print(f"Skipping {path} (no 'query' column)")
            continue
        for q in df["query"].dropna():
            queries.append((path.name, q))

    print(f"Checking {len(queries)} queries across {len(QUERY_FILES)} test-query files "
          f"against {len(ALL_CORPUS_TABLES)} corpus tables...")

    rank1_table_hits = {table: [] for table in ALL_CORPUS_TABLES}
    for source_file, query in queries:
        results, _ = retriever.retrieve_adaptive(query, top_n=1)
        if not results:
            continue
        table = results[0]["metadata"].get("table")
        if table in rank1_table_hits:
            rank1_table_hits[table].append((source_file, query))

    rows = []
    for table in ALL_CORPUS_TABLES:
        hits = rank1_table_hits[table]
        rows.append({
            "table": table,
            "n_rank1_queries": len(hits),
            "covered": len(hits) > 0,
            "example_query": hits[0][1] if hits else "",
            "example_source_file": hits[0][0] if hits else "",
        })

    out = pd.DataFrame(rows).sort_values("n_rank1_queries")
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}\n")
    print(out.to_string(index=False))

    gaps = out[~out["covered"]]
    if len(gaps):
        print(f"\nCOVERAGE GAP: {len(gaps)} table(s) with ZERO rank-1 queries in any test set:")
        for _, row in gaps.iterrows():
            print(f"  - {row['table']}")
    else:
        print("\nNo coverage gaps: every corpus table is rank-1 for at least one test query.")


if __name__ == "__main__":
    main()
