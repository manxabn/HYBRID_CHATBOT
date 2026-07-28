"""
Before implementing a DAT-style (Dynamic Alpha Tuning) learned/continuous
fusion weight, check whether there is actually room for one to help on this
corpus -- the aggregate lambda sweep (Sec.~4.3, fig_lambda_sweep.png) found
NO fixed lambda beats BM25-only on average, but that alone doesn't tell us
whether individual queries disagree about their own best lambda (masked by
averaging) or whether they mostly agree with BM25-only being fine (no real
per-query variance to exploit).

This computes each open-ended query's own ORACLE best lambda (the single
best lambda for THAT query alone, sweeping lambda in {0, 0.05, ..., 1.0})
and compares mean oracle-nDCG@5 against mean nDCG@5 at the single best FIXED
lambda. The gap between these two numbers is a hard upper bound on what any
per-query/continuous weighting scheme (DAT-style or otherwise) could possibly
gain over just using the best fixed lambda -- if the gap is small, building
a learned per-query weight is not worth the engineering effort; if it's
large, it's worth pursuing, and a second check (is the oracle-best-lambda
predictable from cheap query-time features?) would follow.

Retrieval-only, no LLM generation -- safe alongside a concurrently running
Ollama job.

Usage: python scripts/oracle_lambda_ceiling.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline.hybrid_retriever as hr
from measure_ir_metrics import is_relevant, ndcg_at_k

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "oracle_lambda_ceiling.csv"
TOP_K = 10
LAMBDAS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]


def main():
    df = pd.read_csv(QUERIES_PATH)
    open_ended = df[~df["is_entity_heavy"]].reset_index(drop=True)
    print(f"Open-ended queries: {len(open_ended)}/{len(df)}")

    retriever = hr.HybridRetriever()

    rows = []
    for _, r in open_ended.iterrows():
        for lam in LAMBDAS:
            results = retriever.retrieve(r["query"], lam, fusion="linear", top_n=TOP_K)
            rel_ranks = [i for i, c in enumerate(results[:TOP_K])
                         if is_relevant(c, r["query"], str(r["reference_answer"]), False)]
            rows.append({"query_id": r["query_id"], "lambda": lam, "ndcg@5": ndcg_at_k(rel_ranks, 5)})
        print(f"  {r['query_id']} done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    # Best single FIXED lambda (same value for every query): whichever
    # lambda maximizes the mean over all queries.
    by_lambda = out.groupby("lambda")["ndcg@5"].mean()
    best_fixed_lambda = by_lambda.idxmax()
    best_fixed_mean = by_lambda.max()

    # Oracle: best lambda PER QUERY, then average those per-query maxima.
    oracle_per_query = out.groupby("query_id")["ndcg@5"].max()
    oracle_mean = oracle_per_query.mean()

    # Per-query SENSITIVITY to lambda at all (max-min nDCG@5 across the
    # whole sweep, for that query alone): if this is 0, no lambda choice
    # -- fixed, adaptive, or continuous -- can change that query's outcome,
    # so it's uninformative for this question either way. Reported instead
    # of (rather than only) the oracle-best-lambda's std dev, since idxmax
    # ties (common when a query's nDCG@5 is flat across all lambda) would
    # otherwise silently bias the "variance" statistic toward whichever
    # lambda happens to be tried first (0.0) without reflecting real
    # per-query preference.
    range_per_query = out.groupby("query_id")["ndcg@5"].agg(lambda s: s.max() - s.min())
    frac_sensitive = (range_per_query > 1e-9).mean()
    mean_range_if_sensitive = range_per_query[range_per_query > 1e-9].mean() if frac_sensitive > 0 else 0.0

    print(f"\nBest single fixed lambda: {best_fixed_lambda} (mean nDCG@5 = {best_fixed_mean:.4f})")
    print(f"Oracle (best lambda per query): mean nDCG@5 = {oracle_mean:.4f}")
    print(f"Ceiling gain from per-query/continuous weighting over best fixed lambda: {oracle_mean - best_fixed_mean:.4f}")
    print(f"\nFraction of open-ended queries where nDCG@5 varies AT ALL across the full lambda sweep: {frac_sensitive:.1%}")
    print(f"Mean (max-min) nDCG@5 range among those sensitive queries: {mean_range_if_sensitive:.4f}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
