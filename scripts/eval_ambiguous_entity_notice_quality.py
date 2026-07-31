"""
Measures whether this project's ambiguous-entity handling actually helps,
rather than just demonstrating it on one anecdote (the "Rahman" example).
Three conditions, same 55 genuinely-ambiguous queries (data/test_queries_
ambiguous_entity.csv, is_ambiguous=True -- built earlier this project from
real faculty-name collisions, e.g. 16 distinct real "Rahman"s):

  A. no_notice     -- all true candidates in context, but NO ambiguity
                       signal at all (simulates this project's OWN pre-fix
                       behavior: silently confident, arbitrary pick).
  B. flat_notice    -- context + AMBIGUOUS_ENTITY_NOTICE only (this
                       project's first fix: detect ambiguity, ask for
                       clarification, list names).
  C. conditioning   -- context + AMBIGUOUS_ENTITY_NOTICE + the
                       conditioning-attribute hint (this project's second,
                       CondAmbigQA/RAC-adjacent extension: also surface a
                       distinguishing field, e.g. Designation, when the
                       candidates differ on one).

This targets the one component this session's literature benchmark could
NOT find exact top-tier precedent for (closed-KB structured-entity
-collision disambiguation with a surfaced conditioning attribute) -- the
general "ask, don't guess" paradigm is well-established (AmbigQA, EMNLP
2020; Aliannejadi et al., SIGIR 2019), but making THIS SPECIFIC extension
measurable, not just anecdotal, is what turns it from "a fix" into "a
result."

Scoring: an LLM-judge (RAGAS-style, the same pattern already used for
faithfulness in this project) rates each response 0/1 on three criteria:
  - avoids_false_confidence: does NOT present one candidate as the single
    definitive answer.
  - asks_for_clarification: explicitly asks which specific person is meant.
  - offers_disambiguator: gives the user something concrete and easy to
    answer with (a name list counts as a weak disambiguator; a
    distinguishing ATTRIBUTE, e.g. "are they the professor or a lecturer?",
    counts as a strong one) -- this is the criterion condition C is
    specifically expected to win on.

Paired bootstrap significance (same method as every other paired comparison
in this project) across the three conditions on all three criteria.

Usage: python scripts/eval_ambiguous_entity_notice_quality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from pipeline.novel_pipeline import AMBIGUOUS_ENTITY_NOTICE, build_conditioning_hint
from pipeline.ollama_client import generate, MODEL, OLLAMA_URL, post_with_retry, JudgeParseError
import requests

QUERIES_PATH = ROOT / "data" / "test_queries_ambiguous_entity.csv"
OUT_RAW_PATH = ROOT / "results" / "ambiguous_notice_quality_raw.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "ambiguous_notice_quality_summary.csv"
TOP_N = 20  # matches AMBIGUOUS_ENTITY_MAX in novel_pipeline.py

JUDGE_PROMPT = """You are evaluating a chatbot's response to a question about a person whose name matches MULTIPLE distinct real people in a database (a genuinely ambiguous query, e.g. asking about "Rahman" when 16 different faculty share that surname).

Question: {query}
Chatbot's response: {response}

Score the response on three criteria. For each, answer ONLY "yes" or "no".

1. avoids_false_confidence: Does the response AVOID stating one specific answer (like one specific room number or email) as if it were definitively correct, without acknowledging other people also match?
2. asks_for_clarification: Does the response explicitly ask the user which specific person they mean, or otherwise invite clarification?
3. offers_disambiguator: Does the response give the user something concrete and easy to use to narrow it down (e.g. a distinguishing role/title/designation, not just "please specify"), or at minimum list the candidate names so the user could point to one?

Output EXACTLY in this format, nothing else:
avoids_false_confidence: yes/no
asks_for_clarification: yes/no
offers_disambiguator: yes/no"""


def build_context_variant(retriever, query, condition):
    exact_ids = retriever.exact_match_ids(query)
    results, _ = retriever.retrieve_adaptive(query, 0.9, 0.5, top_n=min(len(exact_ids) or TOP_N, TOP_N))
    context_text = "\n\n".join(d["text"] for d in results)

    if condition == "no_notice":
        return context_text
    elif condition == "flat_notice":
        return AMBIGUOUS_ENTITY_NOTICE + "\n\n" + context_text
    elif condition == "conditioning":
        hint = build_conditioning_hint(retriever, exact_ids)
        return AMBIGUOUS_ENTITY_NOTICE + hint + "\n\n" + context_text
    raise ValueError(condition)


def judge_response(query, response):
    prompt = JUDGE_PROMPT.format(query=query, response=response)
    # post_with_retry (2026-07-31): retries on transient Ollama timeout
    # (found real: concurrent-job queueing pileup crashed a sibling script
    # using this same pattern, with zero progress saved) instead of crashing
    # the whole evaluation run on one slow request.
    resp = post_with_retry(
        OLLAMA_URL,
        {"model": MODEL, "prompt": prompt, "stream": False,
         "options": {"temperature": 0.0, "seed": 42, "num_ctx": 1024}},
        timeout=900,
    )
    text = resp.json()["response"].strip().lower()
    scores = {}
    for criterion in ["avoids_false_confidence", "asks_for_clarification", "offers_disambiguator"]:
        line = next((l for l in text.splitlines() if criterion in l), "")
        # 2026-07-31 fix: a missing line is a malformed/truncated response,
        # not a "no" verdict -- raise instead of fabricating a score.
        if not line:
            raise JudgeParseError(f"judge response missing '{criterion}' line", raw_response=text)
        scores[criterion] = 1 if "yes" in line else 0
    return scores


def judge_response_or_fail(query, response, max_attempts=3):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return judge_response(query, response)
        except JudgeParseError as e:
            last_err = e
            print(f"    [parse retry {attempt+1}/{max_attempts}] {e} (raw={e.raw_response!r})", flush=True)
    print(f"    [PARSE FAILURE, giving up after {max_attempts} attempts] {last_err}", flush=True)
    return None


def bootstrap_ci_diff(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), lo, hi, p


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_ambiguous"]].reset_index(drop=True)
    print(f"Evaluating {len(df)} genuinely ambiguous queries x 3 conditions")

    retriever = hr.HybridRetriever()
    conditions = ["no_notice", "flat_notice", "conditioning"]
    rows = []
    n_parse_failed = 0
    for _, r in df.iterrows():
        query = r["query"]
        for cond in conditions:
            context = build_context_variant(retriever, query, cond)
            answer = generate(query, context)
            scores = judge_response_or_fail(query, answer)
            if scores is None:
                n_parse_failed += 1
                rows.append({"query_id": r["query_id"], "condition": cond, "query": query,
                             "answer": answer, "avoids_false_confidence": None,
                             "asks_for_clarification": None, "offers_disambiguator": None,
                             "parse_failed": True})
                print(f"  [{cond}] {r['query_id']}: PARSE FAILED, excluded", flush=True)
                continue
            rows.append({"query_id": r["query_id"], "condition": cond, "query": query,
                         "answer": answer, **scores, "parse_failed": False})
            print(f"  [{cond}] {r['query_id']}: {scores}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_RAW_PATH, index=False)
    print(f"\n{n_parse_failed}/{len(out)} rows FAILED TO PARSE even after retries -- "
          f"excluded from the summary/significance below, not defaulted to any score.")
    out = out[~out["parse_failed"]]

    criteria = ["avoids_false_confidence", "asks_for_clarification", "offers_disambiguator"]
    summary = out.groupby("condition")[criteria].mean()
    summary.to_csv(OUT_SUMMARY_PATH)
    print("\n=== Summary (mean, 0-1) ===")
    print(summary)

    print("\n=== Paired bootstrap significance ===")
    # dropna: a parse failure on just one condition for a given query_id
    # leaves NaN in that cell after pivoting -- drop incomplete query_ids so
    # the paired bootstrap compares genuinely paired arrays, not
    # silently-misaligned ones.
    pivot = {c: out.pivot(index="query_id", columns="condition", values=c)
                 .dropna(subset=["no_notice", "flat_notice", "conditioning"])
             for c in criteria}
    for c in criteria:
        p = pivot[c]
        print(f"  ({c}: n={len(p)} query_ids complete across all 3 conditions)")
        for a, b in [("flat_notice", "no_notice"), ("conditioning", "no_notice"), ("conditioning", "flat_notice")]:
            mean_diff, lo, hi, pval = bootstrap_ci_diff(p[a].values, p[b].values)
            print(f"{c}: {a} vs {b}: diff={mean_diff:.3f} CI=[{lo:.3f},{hi:.3f}] p={pval:.4f} "
                  f"significant={not (lo <= 0 <= hi)}")

    print(f"\nWrote {OUT_RAW_PATH} and {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
