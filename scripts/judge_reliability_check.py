"""
Partial mitigation for "single LLM judge, no human validation of judge
scores" -- the one item from a reviewer-style critique this project cannot
fully resolve tonight (that needs a human, same as the calibration-data
gap). What IS checkable without a human: whether the judge's verdict is a
stable property of the response's actual content, or an artifact of the
exact wording of the judge PROMPT. If a semantically-identical but
differently-worded prompt flips the verdict often, that's real evidence the
judge is unreliable; if verdicts mostly agree, that's a genuine (if
partial) reliability signal worth reporting either way.

Re-scores the SAME already-generated responses (results/ambiguous_notice_
quality_expanded_raw.csv -- no new generation calls, only new judge calls, since the
text being judged doesn't change) with a REWORDED judge prompt: same three
criteria, different phrasing and different question order, to avoid
trivial anchoring on the original prompt's exact structure. Reports percent
agreement and Cohen's kappa per criterion between the original judgments
and this reworded-prompt re-judgment.

Usage: python scripts/judge_reliability_check.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry, JudgeParseError

RAW_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_raw.csv"
OUT_PATH = ROOT / "results" / "judge_reliability_check_expanded.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "judge_reliability_summary_expanded.csv"

# Same three criteria as scripts/eval_ambiguous_entity_notice_quality.py's
# JUDGE_PROMPT, DELIBERATELY reworded (different phrasing, different
# question order) rather than repeated verbatim -- the point is to test
# whether the verdict is stable to surface rewording of an equivalent
# question, not to just re-ask the identical prompt (which would mostly
# just reproduce the same greedy output at temperature=0 and prove nothing).
REWORDED_JUDGE_PROMPT = """A chatbot answered a question about a person whose name matches several DIFFERENT real people in a database (for example, asking about "Rahman" when 16 different staff share that surname).

Question asked: {query}
Chatbot's reply: {response}

Answer three yes/no questions about the reply:

1. requests_clarification: Does the reply ask the user which specific person is meant, or otherwise invite them to clarify?
2. gives_useful_narrowing_info: Does the reply offer the user a practical way to narrow down who is meant -- a distinguishing detail (like a role or title), or at minimum a list of the candidate names?
3. no_overconfident_answer: Does the reply AVOID presenting one specific fact (like a single room number or email) as the definite answer, without acknowledging that other people also match?

Respond in EXACTLY this format, nothing else:
requests_clarification: yes/no
gives_useful_narrowing_info: yes/no
no_overconfident_answer: yes/no"""

# Maps reworded-criterion-name -> original-criterion-name, so the two
# passes can be compared directly despite the deliberate renaming/reordering.
CRITERION_MAP = {
    "no_overconfident_answer": "avoids_false_confidence",
    "requests_clarification": "asks_for_clarification",
    "gives_useful_narrowing_info": "offers_disambiguator",
}


def judge_reworded(query, response):
    prompt = REWORDED_JUDGE_PROMPT.format(query=query, response=response)
    resp = post_with_retry(
        OLLAMA_URL,
        {"model": MODEL, "prompt": prompt, "stream": False,
         "options": {"temperature": 0.0, "seed": 42, "num_ctx": 1024}},
        timeout=900,
    )
    text = resp.json()["response"].strip().lower()
    scores = {}
    for criterion in CRITERION_MAP:
        line = next((l for l in text.splitlines() if criterion in l), "")
        # 2026-07-31 fix: a missing line means the response didn't even
        # mention this criterion -- malformed/truncated output, not a "no"
        # verdict. Defaulting this to 0 previously fabricated a specific
        # score for a response we couldn't actually parse.
        if not line:
            raise JudgeParseError(
                f"reworded judge response missing '{criterion}' line", raw_response=text)
        scores[criterion] = 1 if "yes" in line else 0
    return scores


def judge_reworded_or_fail(query, response, max_attempts=3):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return judge_reworded(query, response)
        except JudgeParseError as e:
            last_err = e
            print(f"    [parse retry {attempt+1}/{max_attempts}] {e} (raw={e.raw_response!r})", flush=True)
    print(f"    [PARSE FAILURE, giving up after {max_attempts} attempts] {last_err}", flush=True)
    return None


def cohens_kappa(a, b):
    """a, b: same-length lists of 0/1 labels from two raters."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if pe == 1.0:
        return 1.0  # both raters constant and identical -- no chance-disagreement possible
    return (po - pe) / (1 - pe)


def main():
    raw = pd.read_csv(RAW_PATH)
    print(f"Re-judging {len(raw)} existing responses with a reworded prompt (no new generation)...")

    rows = []
    n_parse_failed = 0
    for i, r in raw.iterrows():
        reworded_scores = judge_reworded_or_fail(r["query"], r["answer"])
        row = {"query_id": r["query_id"], "condition": r["condition"], "parse_failed": reworded_scores is None}
        if reworded_scores is None:
            n_parse_failed += 1
            for original_name in CRITERION_MAP.values():
                row[f"original__{original_name}"] = r[original_name]
                row[f"reworded__{original_name}"] = None
            rows.append(row)
            print(f"  [{r['condition']}] {r['query_id']}: PARSE FAILED, excluded", flush=True)
            continue
        for reworded_name, original_name in CRITERION_MAP.items():
            row[f"original__{original_name}"] = r[original_name]
            row[f"reworded__{original_name}"] = reworded_scores[reworded_name]
        rows.append(row)
        print(f"  [{r['condition']}] {r['query_id']}: {reworded_scores}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    clean = out[~out["parse_failed"]]
    print(f"\n{n_parse_failed}/{len(out)} rows FAILED TO PARSE even after retries -- "
          f"EXCLUDED from kappa below, not defaulted to any score.")

    summary_rows = []
    for original_name in CRITERION_MAP.values():
        a = clean[f"original__{original_name}"].tolist()
        b = clean[f"reworded__{original_name}"].tolist()
        agreement = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        kappa = cohens_kappa(a, b)
        summary_rows.append({"criterion": original_name, "n": len(a), "n_parse_failed": n_parse_failed,
                              "percent_agreement": round(agreement, 4), "cohens_kappa": round(kappa, 4)})
        print(f"{original_name}: agreement={agreement:.3f} kappa={kappa:.3f}")

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY_PATH, index=False)
    print(f"\nWrote {OUT_PATH} and {OUT_SUMMARY_PATH}")
    print("\nReading kappa: <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 near-perfect "
          "(Landis & Koch 1977 convention). This measures prompt-rewording robustness, NOT agreement with "
          "ground truth or a human -- a high kappa here means the judge is at least self-consistent under "
          "rewording, not that its judgments are correct.")


if __name__ == "__main__":
    main()
