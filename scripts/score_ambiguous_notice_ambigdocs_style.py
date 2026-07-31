"""
Re-scores the ALREADY-GENERATED responses from scripts/eval_ambiguous_
entity_notice_quality.py (results/ambiguous_notice_quality_raw.csv, 55
genuinely-ambiguous queries x 3 conditions, no new generation needed here --
only an additional LLM-judge pass over existing text) using an answer-type
ontology adapted from AmbigDocs (Lee, Ye & Choi, arXiv:2404.12447, 2024) and
its RAMDocs/MADAM-RAG follow-up (Wang, Prasad, Stengel-Eskin & Bansal, COLM
2025, arXiv:2504.13079) -- both independently WebFetch-verified by this
project's own literature-research agent, 2026-07-29. This project's own
homegrown 3-criterion judge (avoids_false_confidence/asks_for_clarification/
offers_disambiguator) already measures something real, but doesn't map onto
a metric anyone else has published a number for. This script adds a SECOND,
independent scoring pass using the closer-to-standard ontology so this
project's own ambiguous-entity result can be framed as "same task shape as
AmbigDocs/RAMDocs, different data" -- a same-metric comparison point, not
just same-paradigm citation.

Ontology (adapted for THIS task -- disambiguating a same-surname query
against a closed faculty roster, not AmbigDocs' cross-document entity
resolution over web text; the adaptation is disclosed, not presented as a
literal reproduction):
  - complete:  the response correctly accounts for ALL true candidates
               (data/test_queries_ambiguous_entity.csv's true_names column
               is the ground truth) -- either by asking the user to
               distinguish among all of them, or by correctly stating
               information for each.
  - partial:   the response accounts for SOME but not all true candidates.
  - merged:    the response states one answer as if it were about a single
               person, but that answer actually blends/conflates fields
               from two or more different true candidates (e.g. combining
               one person's room with another's designation).
  - no_answer: the response does not correctly identify or address any of
               the true candidates (wrong information, or a generic refusal
               that doesn't engage with the actual candidates at all).
  - ambiguous_unclear: the response is too vague to classify into any of
               the above (neither names candidates, asks a clear question,
               nor states any concrete fact).

Usage: python scripts/score_ambiguous_notice_ambigdocs_style.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry, JudgeParseError

RAW_RESPONSES_PATH = ROOT / "results" / "ambiguous_notice_quality_raw.csv"
GROUND_TRUTH_PATH = ROOT / "data" / "test_queries_ambiguous_entity.csv"
OUT_RAW_PATH = ROOT / "results" / "ambiguous_notice_ambigdocs_style_raw.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "ambiguous_notice_ambigdocs_style_summary.csv"

LABELS = ["complete", "partial", "merged", "no_answer", "ambiguous_unclear"]

JUDGE_PROMPT = """A chatbot was asked a question about a person whose name matches MULTIPLE distinct real people in a university database. Here are the ACTUAL distinct people who match (ground truth):
{true_names}

Question: {query}
Chatbot's response: {response}

Classify the response into EXACTLY ONE of these categories:
- complete: the response correctly accounts for ALL of the listed people (either by clearly asking the user to distinguish among all of them, or by correctly giving information about each of them).
- partial: the response accounts for SOME but not all of the listed people (e.g. lists only 2 of 4 real candidates, or answers as if only a subset exist).
- merged: the response states ONE specific answer (like one room number or one email) as if it were about a single person, but that answer actually blends or conflates details from two or more different listed people.
- no_answer: the response does not correctly identify or engage with any of the listed people (wrong information, or a generic refusal that doesn't reference the actual candidates).
- ambiguous_unclear: the response is too vague to classify as any of the above (doesn't name candidates, doesn't ask a clear question, doesn't state any concrete fact).

Output EXACTLY one line, nothing else:
label: <one of complete/partial/merged/no_answer/ambiguous_unclear>"""


def judge_response(query, response, true_names):
    prompt = JUDGE_PROMPT.format(query=query, response=response,
                                  true_names="\n".join(f"- {n}" for n in true_names))
    resp = post_with_retry(
        OLLAMA_URL,
        {"model": MODEL, "prompt": prompt, "stream": False,
         "options": {"temperature": 0.0, "seed": 42, "num_ctx": 1024}},
        timeout=900,
    )
    text = resp.json()["response"].strip().lower()
    for label in LABELS:
        if label in text:
            return label
    # 2026-07-31 fix: previously defaulted to "ambiguous_unclear" here, which
    # silently conflated two different things -- the judge genuinely
    # verdicting "ambiguous_unclear" (which is caught by the loop above,
    # since that literal label string would be found in the text) vs. the
    # response being malformed/truncated and matching NONE of the 5 labels
    # at all. Only the second case reaches this line. Raise instead of
    # fabricating a label for a response we couldn't actually parse.
    raise JudgeParseError(f"no recognized label found in judge response", raw_response=text)


def judge_response_or_fail(query, response, true_names, max_attempts=3):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return judge_response(query, response, true_names)
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
    raw = pd.read_csv(RAW_RESPONSES_PATH)
    gt = pd.read_csv(GROUND_TRUTH_PATH)[["query_id", "true_names"]]
    df = raw.merge(gt, on="query_id", how="left")
    df["true_names_list"] = df["true_names"].str.split("|")
    print(f"Scoring {len(df)} existing responses ({df['query_id'].nunique()} queries x 3 conditions) "
          f"with the AmbigDocs-style answer-type ontology...")

    rows = []
    n_parse_failed = 0
    for i, r in df.iterrows():
        label = judge_response_or_fail(r["query"], r["answer"], r["true_names_list"])
        if label is None:
            n_parse_failed += 1
            rows.append({"query_id": r["query_id"], "condition": r["condition"],
                          "query": r["query"], "label": None, "parse_failed": True})
            print(f"  [{r['condition']}] {r['query_id']}: PARSE FAILED, excluded", flush=True)
            continue
        rows.append({"query_id": r["query_id"], "condition": r["condition"],
                      "query": r["query"], "label": label, "parse_failed": False})
        print(f"  [{r['condition']}] {r['query_id']}: {label}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_RAW_PATH, index=False)
    print(f"\n{n_parse_failed}/{len(out)} rows FAILED TO PARSE even after retries -- "
          f"excluded below, not defaulted to any label.")
    out = out[~out["parse_failed"]]

    summary = out.groupby(["condition", "label"]).size().unstack(fill_value=0)
    summary = summary.reindex(columns=LABELS, fill_value=0)
    summary_frac = summary.div(summary.sum(axis=1), axis=0).round(4)
    summary_frac.to_csv(OUT_SUMMARY_PATH)
    print("\n=== Answer-type distribution by condition (fraction) ===")
    print(summary_frac)

    # complete_rate is the single headline number most directly comparable
    # to AmbigDocs/MADAM-RAG's own "does the system account for all valid
    # answers" framing -- paired bootstrap significance across conditions,
    # same method as every other comparison in this project.
    out["is_complete"] = (out["label"] == "complete").astype(int)
    pivot = out.pivot(index="query_id", columns="condition", values="is_complete")
    pivot = pivot.dropna(subset=["no_notice", "flat_notice", "conditioning"])
    print(f"\n=== Paired bootstrap significance: complete_rate (n={len(pivot)} query_ids complete across conditions) ===")
    for a, b in [("flat_notice", "no_notice"), ("conditioning", "no_notice"), ("conditioning", "flat_notice")]:
        mean_diff, lo, hi, pval = bootstrap_ci_diff(pivot[a].values, pivot[b].values)
        print(f"complete_rate: {a} vs {b}: diff={mean_diff:.3f} CI=[{lo:.3f},{hi:.3f}] p={pval:.4f} "
              f"significant={not (lo <= 0 <= hi)}")

    print(f"\nWrote {OUT_RAW_PATH} and {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
