"""
Direct answer to "is adaptive routing's null result fixable, or is it a real
ceiling?" -- isolates adaptive routing's actual fusion-method choice (RRF at
lambda=0.9) from the UNAMBIGUOUS_MATCH_SCORE=100.0 ceiling in pipeline/
hybrid_retriever.py that forces any single confirmed exact match to rank 0
regardless of which fusion method computed the underlying scores.

Why this diagnostic, not a parameter search: scripts/mcnemar_full_hybrid_
vs_bm25.py already showed 0 discordant pairs between full_hybrid and
bm25_only on every entity-heavy metric under the LIVE configuration -- the
two fusion methods make literally identical top-ranked predictions there.
_score_linear and _score_rrf both add UNAMBIGUOUS_MATCH_SCORE=100.0 (dwarfing
every other term) whenever exactly one exact-match candidate exists, which is
the common case for entity-heavy queries in this corpus. That ceiling, not
adaptive routing's choice of RRF vs. linear fusion, is what's actually
deciding the top rank on those queries -- so testing adaptive routing "as
deployed" can only ever show a null result on this corpus's entity-heavy
subset, independent of whether RRF vs. linear fusion would otherwise differ.

This script asks the direct mechanistic question: with that ceiling
temporarily patched to 0.0 (EXACT_MATCH_BONUS=0.3 still active, so exact
matches still get a real but non-dominant boost), does adaptive routing's
RRF/lambda=0.9 branch produce different top-ranked results than a fixed
full_hybrid linear/lambda=0.5 baseline, on the entity-heavy queries where
their routing decisions actually differ?

This does NOT change deployed behavior (UNAMBIGUOUS_MATCH_SCORE is restored
before the script exits, and this is a standalone diagnostic script, not a
change to hybrid_retriever.py itself) -- it is a one-off measurement to
answer a mechanistic question, not a proposal to weaken a validated,
deployed mechanism for the sake of a paper result. If the two methods still
tie even without the confound, that is real evidence the null is not an
artifact of the ceiling. If they diverge, that is real evidence the ceiling
specifically is what's been masking a genuine effect all along -- either
answer is reported as-is.

Retrieval-only (no Ollama, no generation) -- cheap, GPU-embedding-only.

Usage: python scripts/isolate_adaptive_routing_deconfounded.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_RAW = ROOT / "results" / "adaptive_deconfounded_raw.csv"
OUT_MCNEMAR = ROOT / "results" / "adaptive_deconfounded_mcnemar.csv"
TOP_K = 10

sys.path.insert(0, str(ROOT / "scripts"))
from measure_ir_metrics import is_relevant, recall_at_k, rr, ndcg_at_k  # noqa: E402


def run_config(retriever, df, config_fn, label):
    rows = []
    for _, r in df.iterrows():
        results = config_fn(retriever, r["query"])
        rel_ranks = [i for i, c in enumerate(results[:TOP_K])
                     if is_relevant(c, r["query"], str(r["reference_answer"]), r["is_entity_heavy"])]
        rows.append({
            "query_id": r["query_id"], "config": label,
            "recall@1": recall_at_k(rel_ranks, 1), "recall@3": recall_at_k(rel_ranks, 3),
            "recall@5": recall_at_k(rel_ranks, 5), "mrr": rr(rel_ranks),
            "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
        })
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
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == True].reset_index(drop=True)
    print(f"Entity-heavy queries: {len(df)}", flush=True)

    retriever = hr.HybridRetriever()

    live_value = hr.UNAMBIGUOUS_MATCH_SCORE
    print(f"Live UNAMBIGUOUS_MATCH_SCORE={live_value} -- patching to 0.0 for this diagnostic only", flush=True)
    hr.UNAMBIGUOUS_MATCH_SCORE = 0.0

    try:
        configs = {
            "adaptive_deconfounded": lambda r, q: r.retrieve_adaptive(q, top_n=TOP_K)[0],
            "full_hybrid_deconfounded": lambda r, q: r.retrieve(q, 0.5, fusion="linear", top_n=TOP_K),
        }
        all_results = {}
        for label, fn in configs.items():
            print(f"Running {label}...", flush=True)
            all_results[label] = run_config(retriever, df, fn, label)
    finally:
        hr.UNAMBIGUOUS_MATCH_SCORE = live_value
        print(f"Restored UNAMBIGUOUS_MATCH_SCORE={hr.UNAMBIGUOUS_MATCH_SCORE}", flush=True)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(OUT_RAW, index=False)

    summary = combined.groupby("config")[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4)
    print("\n=== Deconfounded, entity-heavy only (n={}) ===".format(len(df)))
    print(summary)

    adaptive = all_results["adaptive_deconfounded"].set_index("query_id")
    fixed = all_results["full_hybrid_deconfounded"].set_index("query_id")
    common = adaptive.index.intersection(fixed.index)

    rows = []
    for metric in ["recall@1", "recall@3", "recall@5"]:
        a = adaptive.loc[common, metric].values
        b = fixed.loc[common, metric].values
        a_wins, b_wins, n_disc, p = mcnemar(a, b)
        rows.append({
            "metric": metric, "n": len(common),
            "adaptive_only_correct": a_wins, "full_hybrid_only_correct": b_wins,
            "n_discordant": n_disc, "mcnemar_p": round(p, 4),
            "significant_at_0.05": p < 0.05,
        })
        print(f"[{metric}] adaptive-only-correct={a_wins}, full_hybrid-only-correct={b_wins}, "
              f"n_discordant={n_disc}, p={p:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MCNEMAR, index=False)
    print(f"\nWrote {OUT_RAW} and {OUT_MCNEMAR}")

    total_disc = out["n_discordant"].sum()
    if total_disc == 0:
        print("\nConclusion: STILL ZERO discordant pairs even with the exact-match ceiling "
              "removed -- the null is not an artifact of UNAMBIGUOUS_MATCH_SCORE. RRF@0.9 and "
              "linear@0.5 fusion produce the same top-ranked candidate on this corpus's "
              "entity-heavy queries regardless of the ceiling. This is a real ceiling from "
              "EXACT_MATCH_BONUS + this corpus's low candidate-set ambiguity, not a masking "
              "artifact -- adaptive routing's null result holds up under direct mechanistic "
              "removal of the leading suspect, not just repeated re-measurement.")
    else:
        print(f"\nConclusion: {total_disc} discordant pairs appeared once the ceiling was "
              "removed -- the live UNAMBIGUOUS_MATCH_SCORE ceiling was masking a real "
              "difference between fusion methods. See mcnemar_p per metric above for whether "
              "it reaches significance.")


if __name__ == "__main__":
    main()
