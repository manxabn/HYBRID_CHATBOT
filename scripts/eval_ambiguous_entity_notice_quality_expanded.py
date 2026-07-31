"""
Re-runs scripts/eval_ambiguous_entity_notice_quality.py's exact methodology
(3 conditions x LLM-judge x paired bootstrap significance) on the POWERED-UP
ambiguous-entity test set (data/test_queries_ambiguous_entity_expanded.csv,
n=220: 55 genuinely-ambiguous people x 4 field templates, built by
scripts/expand_ambiguous_entity_test.py) instead of the original n=55.

Direct response to a reviewer-style critique: the original conditioning-
vs-flat-notice comparison on offers_disambiguator trended positive
(+0.093) but did not reach significance at n=55 (p=0.088). This reruns the
SAME comparison at 4x the query count (real, distinct questions about the
same real ambiguous people, not repeated/synthetic padding) to see whether
the trend firms up under adequate power, without changing the mechanism,
the judge prompt, or the significance method in any way -- only n changes.

Usage: python scripts/eval_ambiguous_entity_notice_quality_expanded.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from eval_ambiguous_entity_notice_quality import build_context_variant, judge_response, bootstrap_ci_diff
from pipeline.ollama_client import generate

QUERIES_PATH = ROOT / "data" / "test_queries_ambiguous_entity_expanded.csv"
OUT_RAW_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_raw.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_summary.csv"


def main():
    df = pd.read_csv(QUERIES_PATH)
    print(f"Evaluating {len(df)} expanded ambiguous queries x 3 conditions "
          f"({df['name_token'].nunique()} people x {df['field_asked'].nunique()} field templates)")

    retriever = hr.HybridRetriever()
    conditions = ["no_notice", "flat_notice", "conditioning"]
    rows = []
    for i, r in df.iterrows():
        query = r["query"]
        for cond in conditions:
            context = build_context_variant(retriever, query, cond)
            answer = generate(query, context)
            scores = judge_response(query, answer)
            rows.append({"query_id": r["query_id"], "condition": cond, "query": query,
                         "field_asked": r["field_asked"], "answer": answer, **scores})
            print(f"  [{cond}] {r['query_id']} ({r['field_asked']}): {scores}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"--- {i+1}/{len(df)} people-questions done ---", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_RAW_PATH, index=False)

    criteria = ["avoids_false_confidence", "asks_for_clarification", "offers_disambiguator"]
    summary = out.groupby("condition")[criteria].mean()
    summary.to_csv(OUT_SUMMARY_PATH)
    print("\n=== Summary (mean, 0-1), n=220 ===")
    print(summary)

    print("\n=== By field asked ===")
    print(out.groupby(["field_asked", "condition"])[criteria].mean())

    print("\n=== Paired bootstrap significance (n=220) ===")
    pivot = {c: out.pivot(index="query_id", columns="condition", values=c) for c in criteria}
    for c in criteria:
        p = pivot[c]
        for a, b in [("flat_notice", "no_notice"), ("conditioning", "no_notice"), ("conditioning", "flat_notice")]:
            mean_diff, lo, hi, pval = bootstrap_ci_diff(p[a].values, p[b].values)
            print(f"{c}: {a} vs {b}: diff={mean_diff:.3f} CI=[{lo:.3f},{hi:.3f}] p={pval:.4f} "
                  f"significant={not (lo <= 0 <= hi)}")

    print(f"\nWrote {OUT_RAW_PATH} and {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
