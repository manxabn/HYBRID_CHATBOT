"""
Third attempt at negative-label mining for conformal calibration, after two
failed attempts:
  1. Organic Out-of-Scope-query mining -- failed, contaminated (the system
     mostly refuses/redirects correctly rather than hallucinating).
  2. Adversarial same-category context-swap -- failed, only 1/80 usable
     example (even mismatched-but-real context didn't reliably produce
     confident wrong answers -- the model tends to notice the mismatch).

Root-cause insight behind this third attempt: conformal's own NLI-based
claim scoring checks "is this claim ENTAILED BY THE CONTEXT ACTUALLY
GIVEN" -- not "is this claim true in the real world." That means a genuine
negative example doesn't require a WRONG context at all -- it only
requires an INCOMPLETE one. If the specific answering fact is surgically
removed from an otherwise-correct, real context (truncating the source
Answer text before the actual answering clause) and the model still states
that specific fact confidently, the claim is provably NOT entailed by what
it was actually given -- a real, verifiable "confident but ungrounded"
claim, regardless of whether the stated fact happens to be true. This
mirrors a realistic near-miss retrieval failure mode (the right general
area retrieved, the specific answering sentence missing or truncated) more
closely than either of the first two attempts, and does not depend on the
model failing to notice an obvious mismatch -- it only depends on the
model over-extrapolating from partial information, a much more common
failure mode.

Method: for each open-ended anchor query, take its own TRUE context chunk
(Question: ...\\nAnswer: ...) and truncate the Answer to its first N words
(N chosen per-row to cut roughly halfway through, before the specific
answering clause where possible -- checked heuristically, not guaranteed
exact). Generate an answer to the anchor's own question using this
truncated context. A claim is a valid negative example if it is NOT a
hedge/refusal AND has near-zero lexical overlap with the REMOVED portion
of the answer (i.e., it does not simply repeat words that were still
present in the truncated context) AND is specific enough to check (passes
the same is_specific_fact filter used in the second attempt) -- meaning it
states a concrete fact that could only have come from outside what was
actually given.

Usage: python scripts/build_conformal_truncated_context_negatives.py
"""

import random
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.ollama_client import generate
from pipeline.conformal_abstention import decompose_claims
from fix_conformal_negative_labels import is_specific_fact, is_refusal_or_opinion
from build_conformal_calibration_labels import claim_overlap_ratio

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "results" / "conformal_truncated_context_negative_labels.csv"
SEED = 42
N_QUERIES = 100
TRUNCATE_FRACTION = 0.5  # keep roughly the first half of the Answer, by word count


def load_qa_pool():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT Question, Answer FROM EnglishQA WHERE Split='test' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND Question IS NOT NULL AND Answer IS NOT NULL"
    )
    rows = [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    conn.close()
    # Only rows with a long enough answer that truncation actually removes
    # substantive content, not a one-line answer where "half" is nothing.
    return [(q, a) for q, a in rows if len(a.split()) >= 12]


def truncate_answer(answer: str, frac: float = TRUNCATE_FRACTION):
    words = answer.split()
    cut = max(3, int(len(words) * frac))
    kept = " ".join(words[:cut])
    removed = " ".join(words[cut:])
    return kept, removed


def main():
    pool = load_qa_pool()
    print(f"QA pool with long-enough answers (Split=test, answerable): {len(pool)} rows")
    rng = random.Random(SEED)
    rng.shuffle(pool)
    sample = pool[:N_QUERIES]
    print(f"Sampled {len(sample)} queries for truncated-context negative mining")

    rows = []
    n_hedge = n_kept = n_ambiguous = 0
    for i, (query, answer) in enumerate(sample):
        kept_text, removed_text = truncate_answer(answer)
        truncated_context = f"Question: {query}\nAnswer: {kept_text}..."
        generated = generate(query, truncated_context)
        for claim in decompose_claims(generated):
            if is_refusal_or_opinion(claim):
                n_hedge += 1
                continue
            if not is_specific_fact(claim):
                n_ambiguous += 1
                continue
            overlap_with_kept = claim_overlap_ratio(claim, kept_text)
            overlap_with_removed = claim_overlap_ratio(claim, removed_text)
            # A valid negative: the claim's specific content is NOT explained
            # by what the model actually saw (low overlap with kept_text),
            # regardless of whether it happens to match the removed part
            # (that would mean the model got lucky/used parametric
            # knowledge -- still not grounded in the GIVEN context, so still
            # a valid "not entailed by context" example for calibration
            # purposes, per this script's own module docstring).
            if overlap_with_kept < 0.4:
                rows.append({"claim": claim, "context": truncated_context, "is_correct": False,
                             "source": "truncated_context", "anchor_query": query,
                             "overlap_with_kept_context": round(overlap_with_kept, 3),
                             "overlap_with_removed_answer": round(overlap_with_removed, 3)})
                n_kept += 1
            else:
                n_ambiguous += 1
        print(f"  [{i+1}/{len(sample)}] negatives_so_far={n_kept}", flush=True)

    print(f"\nNegative labels collected: {n_kept} "
          f"(skipped {n_hedge} correct hedges/refusals, {n_ambiguous} ambiguous/grounded-in-kept-context)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} truncated-context negative labels to {OUT_PATH}")
    if len(out):
        print("\nSample of collected negatives:")
        for _, r in out.head(10).iterrows():
            print(f"  - claim: {str(r['claim'])[:130]!r}")
            print(f"    overlap_with_kept={r['overlap_with_kept_context']} "
                  f"overlap_with_removed={r['overlap_with_removed_answer']}")


if __name__ == "__main__":
    main()
