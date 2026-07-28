"""
Builds a test set of genuinely ambiguous and newly-unambiguous-but-partial
faculty-name queries, made possible by two fixes in pipeline/hybrid_
retriever.py (2026-07-28): (1) a possessive-normalization bug ("Kaykobad's"
was mangled to "kaykobads" and matched nothing), and (2) a new token-level
name index that recognizes a single distinctive name fragment ("Kaykobad",
"Anika") rather than requiring the full stored name in the query.

This directly fills a gap flagged by two earlier analyses this session
(scripts/sweep_rrf_k.py, scripts/isolate_adaptive_routing.py): every
entity-heavy query in the main 200-query test set happens to resolve to
exactly one exact match, so RRF fusion and adaptive routing have never been
exercised on a genuinely ambiguous entity resolution -- there was nothing
for the fusion method to decide. Partial-name queries are exactly the
regime where that finally happens: a name fragment shared by multiple
faculty members produces multiple exact-match candidates, and how they get
ranked relative to each other is a real fusion-method question for the
first time.

Ground truth: for AMBIGUOUS queries (shared name fragment), the correct
answer is the SET of doc_ids matching that fragment (derived directly from
the live faculty_name_token_index, not hand-typed) -- there is no single
correct top-1 answer by construction, so this test measures whether the
true candidate set is *recovered* (recall over the full ambiguous set), not
top-1 accuracy. For UNAMBIGUOUS partial-name queries (fragment matches
exactly one person), the correct answer is that one person, same as any
other unambiguous-match test.

Usage: python scripts/build_ambiguous_entity_test.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever

OUT_PATH = ROOT / "data" / "test_queries_ambiguous_entity.csv"


def main():
    r = HybridRetriever()

    rows = []
    qid = 0
    for token, doc_ids in sorted(r.faculty_name_token_index.items()):
        unique_ids = sorted(set(doc_ids))
        names = [r.corpus[d]["metadata"].get("Name") for d in unique_ids]
        qid += 1
        query_id = f"AMB{qid:03d}"
        is_ambiguous = len(unique_ids) > 1
        rows.append({
            "query_id": query_id,
            "query": f"What is {token.capitalize()}'s office room?",
            "name_token": token,
            "is_ambiguous": is_ambiguous,
            "n_true_candidates": len(unique_ids),
            "true_doc_ids": "|".join(unique_ids),
            "true_names": "|".join(n for n in names if n),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} queries to {OUT_PATH}")
    print(f"  Ambiguous (>1 candidate): {df['is_ambiguous'].sum()}")
    print(f"  Unambiguous (exactly 1, only reachable via the token-index fix): {(~df['is_ambiguous']).sum()}")
    print(f"  Ambiguity distribution: {df[df['is_ambiguous']]['n_true_candidates'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
