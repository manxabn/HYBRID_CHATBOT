"""
Redesigns the avoids_false_confidence criterion after finding it's nearly
unreliable (Cohen's kappa=0.114 across a reworded-prompt paraphrase
-invariance check, scripts/judge_reliability_check.py) -- barely above
chance, while the other two criteria (asks_for_clarification kappa=0.984,
offers_disambiguator kappa=0.656) are fine.

Root-cause hypothesis: avoids_false_confidence is phrased as an ABSENCE
judgment ("does the response AVOID stating one answer confidently") --
this requires the judge to first parse what would count as "confident,"
then judge its absence, a harder and more phrasing-sensitive task than a
direct presence check. The other two reliable criteria are both direct
presence checks ("does it ask", "does it offer").

Fix tested here: decompose into a concrete EXTRACTION task instead of one
holistic yes/no judgment --
  1. Which specific candidate name(s), if any, does the response present
     as ITS answer? (a listing/extraction task, not a holistic judgment)
  2. Does the response explicitly acknowledge that multiple people match?
     (still a presence check, like the two reliable criteria)
avoids_false_confidence is then computed DETERMINISTICALLY:
  fails (avoids_false_confidence=No) iff exactly one name was extracted
  AND no multi-match acknowledgment was found.

This is validated the same way the original problem was found -- rerun
the SAME reworded-prompt paraphrase-invariance check on the NEW decomposed
judge and compare kappa directly, on a real subsample, not asserted to
work. If it doesn't actually improve kappa, that is reported too.

Usage: python scripts/improve_avoids_false_confidence_judge.py
"""

import random
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry, JudgeParseError

RAW_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_raw.csv"
OUT_PATH = ROOT / "results" / "improved_afc_judge_check.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "improved_afc_judge_summary.csv"
N_SAMPLE = 150
SEED = 42

EXTRACT_PROMPT_A = """A chatbot answered a question about a person whose name matches several DIFFERENT real people in a database.

Question: {query}
Chatbot's reply: {response}

Answer two questions about the reply, EXACTLY in this format, nothing else:

names_as_answer: <comma-separated list of specific person names the reply states AS ITS ANSWER (e.g. gives their room/email/designation as if confirmed), or "none" if it names no one this way>
acknowledges_multiple: yes/no (does the reply explicitly say or imply that more than one person matches, anywhere in the text?)"""

# Same two questions, reworded + reordered, for the paraphrase-invariance
# check -- NOT the identical prompt (which would trivially reproduce the
# same output at temperature=0).
EXTRACT_PROMPT_B = """Below is a chatbot's reply to a question about someone whose name several different real people share.

User's question: {query}
Reply: {response}

Respond with EXACTLY these two lines, nothing else:

acknowledges_multiple: yes/no -- does the reply anywhere state or imply that more than one person could match?
names_as_answer: <the specific name(s), comma-separated, that the reply commits to as ITS answer (giving a room/email/designation/etc. as if settled) -- or "none" if it doesn't commit to any specific name that way>"""


def extract(prompt_template, query, response):
    prompt = prompt_template.format(query=query, response=response)
    # post_with_retry (2026-07-31): a real ReadTimeout crashed a full run of
    # this exact function with zero progress saved, caused by running
    # several Ollama-dependent scripts concurrently (this project's own
    # parallel-ablation pattern) -- transient queueing contention, not a
    # broken Ollama instance. Retries with backoff before giving up.
    resp = post_with_retry(
        OLLAMA_URL,
        {"model": MODEL, "prompt": prompt, "stream": False,
         "options": {"temperature": 0.0, "seed": 42, "num_ctx": 1024}},
        timeout=900,
    )
    text = resp.json()["response"].strip().lower()
    names_line = next((l for l in text.splitlines() if "names_as_answer" in l), "")
    ack_line = next((l for l in text.splitlines() if "acknowledges_multiple" in l), "")
    # Explicit failure detection (2026-07-31 fix): if EITHER expected field
    # marker is missing, the response is malformed/incomplete -- raise
    # rather than silently defaulting n_names=0/acknowledges=False, which
    # would fabricate a specific, wrong-by-construction judgment for a
    # response we couldn't actually parse. Callers must catch this and
    # record the row as an explicit failure, not a fake score.
    if not names_line or not ack_line:
        raise JudgeParseError(
            f"could not find expected fields in judge response (names_line={names_line!r}, "
            f"ack_line={ack_line!r})", raw_response=text)
    names_str = names_line.split(":", 1)[1].strip() if ":" in names_line else ""
    n_names = 0 if (not names_str or "none" in names_str) else len([n for n in names_str.split(",") if n.strip()])
    acknowledges_multiple = "yes" in ack_line
    avoids_false_confidence = not (n_names == 1 and not acknowledges_multiple)
    return {"n_names_as_answer": n_names, "acknowledges_multiple": int(acknowledges_multiple),
            "avoids_false_confidence": int(avoids_false_confidence)}


def cohens_kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def extract_or_fail(prompt_template, query, response, max_attempts=3):
    """Retries extract() on JudgeParseError (a malformed/incomplete
    response is plausibly a one-off fluke, not necessarily systematic) --
    returns None (an explicit, honest failure marker) if all attempts still
    fail to parse, rather than ever fabricating a default score."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return extract(prompt_template, query, response)
        except JudgeParseError as e:
            last_err = e
            print(f"    [parse retry {attempt+1}/{max_attempts}] {e} (raw={e.raw_response!r})", flush=True)
    print(f"    [PARSE FAILURE, giving up after {max_attempts} attempts] {last_err}", flush=True)
    return None


def main():
    df = pd.read_csv(RAW_PATH)
    rng = random.Random(SEED)
    idx = rng.sample(range(len(df)), min(N_SAMPLE, len(df)))
    sample = df.iloc[idx].reset_index(drop=True)
    print(f"Testing decomposed avoids_false_confidence judge on n={len(sample)} "
          f"(subsample, not the full 660 -- validation pass first)")

    rows = []
    n_parse_failed = 0
    for i, r in sample.iterrows():
        a_result = extract_or_fail(EXTRACT_PROMPT_A, r["query"], r["answer"])
        b_result = extract_or_fail(EXTRACT_PROMPT_B, r["query"], r["answer"])
        if a_result is None or b_result is None:
            n_parse_failed += 1
            rows.append({
                "query_id": r["query_id"], "condition": r["condition"],
                "old_judge_score": int(r["avoids_false_confidence"]),
                "new_judge_A": None, "new_judge_B": None,
                "A_n_names": None, "A_acknowledges": None, "B_n_names": None, "B_acknowledges": None,
                "parse_failed": True,
            })
            print(f"  [{i+1}/{len(sample)}] {r['query_id']}: PARSE FAILED, excluded from kappa calc", flush=True)
            continue
        rows.append({
            "query_id": r["query_id"], "condition": r["condition"],
            "old_judge_score": int(r["avoids_false_confidence"]),
            "new_judge_A": a_result["avoids_false_confidence"],
            "new_judge_B": b_result["avoids_false_confidence"],
            "A_n_names": a_result["n_names_as_answer"], "A_acknowledges": a_result["acknowledges_multiple"],
            "B_n_names": b_result["n_names_as_answer"], "B_acknowledges": b_result["acknowledges_multiple"],
            "parse_failed": False,
        })
        print(f"  [{i+1}/{len(sample)}] {r['query_id']} old={int(r['avoids_false_confidence'])} "
              f"new_A={a_result['avoids_false_confidence']} new_B={b_result['avoids_false_confidence']}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    clean = out[~out["parse_failed"]]
    print(f"\n{n_parse_failed}/{len(out)} rows FAILED TO PARSE even after retries -- "
          f"EXCLUDED from the kappa calculation below, not defaulted to any score.")

    a = clean["new_judge_A"].tolist()
    b = clean["new_judge_B"].tolist()
    old = clean["old_judge_score"].tolist()
    agreement_ab = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    kappa_ab = cohens_kappa(a, b)
    agreement_a_old = sum(1 for x, y in zip(a, old) if x == y) / len(a)

    summary = pd.DataFrame([{
        "comparison": "new_decomposed_judge: prompt A vs prompt B (reliability of the FIX)",
        "n": len(clean), "n_parse_failed": n_parse_failed,
        "raw_agreement": round(agreement_ab, 4), "cohens_kappa": round(kappa_ab, 4),
    }, {
        "comparison": "new_judge_A vs OLD holistic judge (how much the verdict actually changed)",
        "n": len(clean), "n_parse_failed": n_parse_failed,
        "raw_agreement": round(agreement_a_old, 4), "cohens_kappa": None,
    }])
    summary.to_csv(OUT_SUMMARY_PATH, index=False)

    print(f"\n=== Decomposed judge reliability (prompt A vs B), n={len(clean)} (clean, parse failures excluded) ===")
    print(f"raw_agreement={agreement_ab:.4f} cohens_kappa={kappa_ab:.4f} "
          f"(old holistic judge was kappa=0.114 on the same kind of check)")
    print(f"\nFor reference, new_judge_A agrees with the OLD holistic judge {agreement_a_old:.4f} of the time "
          f"(shows how much the verdict shifted, not a reliability number itself)")
    print(f"\nWrote {OUT_PATH} and {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
