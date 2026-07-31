"""
Second attempt at negative-label mining for conformal calibration, after the
first (Out-of-Scope-query mining, scripts/build_conformal_calibration_
labels.py) failed on inspection: the system mostly refuses/redirects
correctly on genuinely unanswerable questions, so organic mining found too
few real hallucinations to calibrate against.

This targets the actual failure mode conformal abstention exists to catch:
a NEAR-MISS retrieval bringing back a topically-similar but factually wrong
chunk. Simulated directly and verifiably: for an open-ended anchor query,
deliberately swap in a "confusor" chunk -- a REAL corpus Q/A pair from the
SAME Category but a DIFFERENT Answer (so it's topically similar, not an
absurd mismatch a model would trivially reject) -- as the retrieved
context, then generate an answer to the anchor's own question using that
wrong context.

Ground truth is structural, not judged: the anchor's true reference_answer
and the confusor's actual answer are both known text, by construction
different (same category, different Answer, and not from the same
(Category, Answer) paraphrase cluster). A generated claim is labeled:
  - negative (is_correct=False) if it lexically overlaps with the
    CONFUSOR's answer (the model repeated/adapted the wrong context's
    specific facts as if they answered the real question) -- a verifiable
    hallucination, not a judgment call.
  - excluded if it's a hedge/refusal (correctly declining given
    insufficient/wrong context -- not a failure case).
  - excluded (ambiguous) otherwise -- e.g. overlaps with the TRUE reference
    despite being given wrong context (parametric knowledge bleed-through,
    a different and murkier case not worth mislabeling either way).

Usage: python scripts/build_conformal_adversarial_negatives.py
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
OUT_PATH = ROOT / "results" / "conformal_adversarial_negative_labels.csv"
SEED = 42
N_PAIRS = 80


def load_qa_pool():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT Question, Answer, Category FROM EnglishQA WHERE Split='test' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND Question IS NOT NULL AND Answer IS NOT NULL"
    )
    rows = [(q.strip(), a.strip(), c) for q, a, c in cur.fetchall() if q.strip() and a.strip()]
    conn.close()
    return rows


def build_confusor_pairs(pool, n_pairs, seed=SEED):
    rng = random.Random(seed)
    by_category = {}
    for q, a, c in pool:
        by_category.setdefault(c, []).append((q, a))

    pairs = []
    categories = [c for c, items in by_category.items() if len(items) >= 2]
    rng.shuffle(categories)
    for cat in categories:
        items = by_category[cat][:]
        rng.shuffle(items)
        for i in range(0, len(items) - 1, 2):
            (anchor_q, anchor_a), (confusor_q, confusor_a) = items[i], items[i + 1]
            if anchor_a.strip().lower() == confusor_a.strip().lower():
                continue  # same underlying fact (paraphrase pair) -- not a real confusor
            pairs.append({"anchor_query": anchor_q, "anchor_answer": anchor_a,
                          "confusor_query": confusor_q, "confusor_answer": confusor_a, "category": cat})
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def main():
    pool = load_qa_pool()
    print(f"QA pool (Split=test, answerable): {len(pool)} rows")
    pairs = build_confusor_pairs(pool, N_PAIRS)
    print(f"Built {len(pairs)} same-category confusor pairs")

    rows = []
    n_hedge = n_wrong_context_used = n_ambiguous = 0
    for i, p in enumerate(pairs):
        wrong_context = f"Question: {p['confusor_query']}\nAnswer: {p['confusor_answer']}"
        answer = generate(p["anchor_query"], wrong_context)
        for claim in decompose_claims(answer):
            if is_refusal_or_opinion(claim):
                n_hedge += 1
                continue
            overlap_wrong = claim_overlap_ratio(claim, p["confusor_answer"])
            overlap_true = claim_overlap_ratio(claim, p["anchor_answer"])
            if overlap_wrong >= 0.5 and overlap_true < 0.3:
                rows.append({"claim": claim, "context": wrong_context, "is_correct": False,
                             "source": "adversarial_context_swap",
                             "anchor_query": p["anchor_query"], "category": p["category"]})
                n_wrong_context_used += 1
            else:
                n_ambiguous += 1
        print(f"  [{i+1}/{len(pairs)}] category={p['category']!r} "
              f"claims_added_so_far={len(rows)}", flush=True)

    print(f"\nNegative labels collected: {len(rows)} "
          f"(skipped {n_hedge} correct hedges/refusals, {n_ambiguous} ambiguous/no-clear-overlap)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} adversarial negative labels to {OUT_PATH}")
    if len(out):
        print("\nSample of collected negatives (should look like genuine wrong-context hallucinations):")
        for _, r in out.head(10).iterrows():
            print(f"  - {str(r['claim'])[:150]!r}")


if __name__ == "__main__":
    main()
