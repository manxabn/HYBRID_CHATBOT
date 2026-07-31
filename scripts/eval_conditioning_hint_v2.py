"""
Tests an IMPROVED conditioning-hint phrasing against the original, after
scripts/eval_ambiguous_entity_notice_quality_expanded.py found (n=220) that
the original conditioning hint is NOT significantly better than the
simpler flat_notice on any of the 3 criteria.

Hypothesis for why the original underperforms: its phrasing is
conditional/passive ("if the user's phrasing already narrows it down by
X, use that") -- it tells the model to check for something rather than
directly instructing it to OFFER the distinguishing attribute as part of
its own clarifying question. build_conditioning_hint_v2 below is more
directive: it gives the model a concrete question template using the
actual distinguishing values, e.g. "Which one do you mean: Professor or
Lecturer?" instead of leaving the model to figure out how to use the
field.

Compares 4 conditions this time (no_notice, flat_notice, conditioning
[original], conditioning_v2 [improved]) on the SAME 220-query expanded
ambiguous-entity set, reusing the exact same judge/scoring/significance
methodology as the existing evaluation scripts.

Checkpointed (added 2026-07-31 after this exact job was killed twice by
infrastructure issues outside this script -- a session interruption and a
transient Ollama 500 -- each restart previously wasted 100% of prior
progress since output was only written once at the very end): if
OUT_RAW_PATH already exists from a prior partial run, already-completed
(query_id, condition) pairs are skipped and new rows are appended, not
overwritten. Each row is flushed to disk immediately after being computed,
not batched, so a crash at row N never loses rows before N. Safe to
Ctrl-C and re-run this exact command to resume.

Usage: python scripts/eval_conditioning_hint_v2.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline.hybrid_retriever as hr
from pipeline.novel_pipeline import AMBIGUOUS_ENTITY_NOTICE, _CONDITIONING_FIELDS
from pipeline.ollama_client import generate
from eval_ambiguous_entity_notice_quality import judge_response_or_fail, bootstrap_ci_diff

QUERIES_PATH = ROOT / "data" / "test_queries_ambiguous_entity_expanded.csv"
OUT_RAW_PATH = ROOT / "results" / "conditioning_hint_v2_raw.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "conditioning_hint_v2_summary.csv"
TOP_N = 20


def build_conditioning_hint_v2(retriever, doc_ids) -> str:
    """Improved phrasing: gives the model a concrete question template using
    the actual distinguishing values, rather than a conditional instruction
    to check the user's own phrasing."""
    records = [retriever.corpus.get(doc_id, {}).get("metadata", {}) for doc_id in doc_ids]
    records = [r for r in records if r]
    if len(records) < 2:
        return ""
    for field in _CONDITIONING_FIELDS:
        values = {r.get(field) for r in records if r.get(field)}
        if len(values) > 1:
            values_sorted = sorted(values)
            values_str = " or ".join(values_sorted) if len(values_sorted) <= 3 else ", ".join(values_sorted)
            return (f" These candidates have different {field} values: {', '.join(values_sorted)}. "
                     f"Use this directly in your clarifying question -- for example, ask "
                     f"\"Which one do you mean: {values_str}?\" instead of only asking for a name.")
    return ""


def build_context_variant(retriever, query, condition):
    exact_ids = retriever.exact_match_ids(query)
    results, _ = retriever.retrieve_adaptive(query, 0.9, 0.5, top_n=min(len(exact_ids) or TOP_N, TOP_N))
    context_text = "\n\n".join(d["text"] for d in results)

    if condition == "no_notice":
        return context_text
    elif condition == "flat_notice":
        return AMBIGUOUS_ENTITY_NOTICE + "\n\n" + context_text
    elif condition == "conditioning":
        from pipeline.novel_pipeline import build_conditioning_hint
        hint = build_conditioning_hint(retriever, exact_ids)
        return AMBIGUOUS_ENTITY_NOTICE + hint + "\n\n" + context_text
    elif condition == "conditioning_v2":
        hint = build_conditioning_hint_v2(retriever, exact_ids)
        return AMBIGUOUS_ENTITY_NOTICE + hint + "\n\n" + context_text
    raise ValueError(condition)


FIELDNAMES = ["query_id", "condition", "query", "field_asked", "answer",
              "avoids_false_confidence", "asks_for_clarification", "offers_disambiguator", "parse_failed"]


def main():
    df = pd.read_csv(QUERIES_PATH)
    conditions = ["no_notice", "flat_notice", "conditioning", "conditioning_v2"]

    completed = set()
    resuming = OUT_RAW_PATH.exists()
    if resuming:
        prior = pd.read_csv(OUT_RAW_PATH)
        completed = set(zip(prior["query_id"], prior["condition"]))
        print(f"Resuming: {len(completed)} (query_id, condition) pairs already done in {OUT_RAW_PATH}")

    print(f"Evaluating {len(df)} expanded ambiguous queries x 4 conditions "
          f"(adding conditioning_v2 to the existing 3)")

    retriever = hr.HybridRetriever()
    out_file = open(OUT_RAW_PATH, "a" if resuming else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if not resuming:
        writer.writeheader()

    n_parse_failed = 0
    for i, r in df.iterrows():
        query = r["query"]
        for cond in conditions:
            if (r["query_id"], cond) in completed:
                continue
            context = build_context_variant(retriever, query, cond)
            answer = generate(query, context)
            scores = judge_response_or_fail(query, answer)
            if scores is None:
                n_parse_failed += 1
                row = {"query_id": r["query_id"], "condition": cond, "query": query,
                       "field_asked": r.get("field_asked", ""), "answer": answer,
                       "avoids_false_confidence": None, "asks_for_clarification": None,
                       "offers_disambiguator": None, "parse_failed": True}
                print(f"  [{cond}] {r['query_id']}: PARSE FAILED, excluded", flush=True)
            else:
                row = {"query_id": r["query_id"], "condition": cond, "query": query,
                       "field_asked": r.get("field_asked", ""), "answer": answer, **scores,
                       "parse_failed": False}
                print(f"  [{cond}] {r['query_id']}: {scores}", flush=True)
            writer.writerow(row)
            out_file.flush()
        if (i + 1) % 20 == 0:
            print(f"--- {i+1}/{len(df)} done ---", flush=True)
    out_file.close()

    out = pd.read_csv(OUT_RAW_PATH)
    n_parse_failed = int(out["parse_failed"].sum())
    print(f"\n{n_parse_failed}/{len(out)} rows FAILED TO PARSE even after retries -- "
          f"excluded from the summary/significance below, not defaulted to any score.")
    out = out[~out["parse_failed"]]

    criteria = ["avoids_false_confidence", "asks_for_clarification", "offers_disambiguator"]
    summary = out.groupby("condition")[criteria].mean()
    summary.to_csv(OUT_SUMMARY_PATH)
    print("\n=== Summary (mean, 0-1), n=220, 4 conditions ===")
    print(summary)

    print("\n=== Paired bootstrap significance: conditioning_v2 vs the other 3 ===")
    # dropna: a parse failure on just one condition for a query_id leaves
    # NaN in that cell after pivoting -- drop incomplete query_ids so the
    # paired bootstrap compares genuinely paired arrays.
    pivot = {c: out.pivot(index="query_id", columns="condition", values=c)
                 .dropna(subset=["no_notice", "flat_notice", "conditioning", "conditioning_v2"])
             for c in criteria}
    for c in criteria:
        p = pivot[c]
        print(f"  ({c}: n={len(p)} query_ids complete across all 4 conditions)")
        for a, b in [("conditioning_v2", "no_notice"), ("conditioning_v2", "flat_notice"),
                     ("conditioning_v2", "conditioning")]:
            mean_diff, lo, hi, pval = bootstrap_ci_diff(p[a].values, p[b].values)
            print(f"{c}: {a} vs {b}: diff={mean_diff:.3f} CI=[{lo:.3f},{hi:.3f}] p={pval:.4f} "
                  f"significant={not (lo <= 0 <= hi)}")

    print(f"\nWrote {OUT_RAW_PATH} and {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
