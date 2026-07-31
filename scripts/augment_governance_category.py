"""
Proposed augmentation for EnglishQA's smallest, most fragile category,
"University Facts & Governance" (66 rows total, and -- the specific problem
this addresses -- only 2 of its 22 unique underlying facts have ANY
representation in the val split at all, each with just 1 row). Found during
this session's dataset-revision review, not assumed.

Does NOT invent new facts about BRAC University (a real fabrication risk
this project's standing rule forbids) -- generates new PARAPHRASES of the
22 already-existing, already-verified (Answer) facts, the same "expand
phrasing diversity without inventing new source content" principle already
used and disclosed for scripts/expand_crosslingual_stress_test.py, applied
here to English paraphrase generation instead of Banglish rephrasing.

Diversity check, following the Selective Few-Random-Shot Augmentation
principle (Hu et al., Applied Intelligence -- independently verified this
session): over-generate candidates, then keep only those that are NOT
near-duplicates of each other or of the existing questions for that fact
(cosine similarity < DIVERSITY_THRESHOLD against the currently-deployed
embedding model), rather than accepting every generated paraphrase --
directly guards against the "200 generated pairs collapse into 5 semantic
clusters" failure mode this session's codebase audit flagged as a risk of
naive templated augmentation.

OUTPUT IS PROPOSED, NOT APPLIED: writes to data/governance_augmentation_
proposed.csv only -- does NOT touch knowledge_base.db, does NOT insert into
EnglishQA, does NOT change any existing evaluation. Modifying the actual
dataset every other result in this project is measured against is a
higher-blast-radius action reserved for the user's explicit review and
decision, consistent with how this session has handled every other
higher-blast-radius action (legacy file moves, git init) -- ask first,
don't act unilaterally.

Usage: python scripts/augment_governance_category.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry
from pipeline.chroma_embedding import Chroma1xEmbeddingFunction

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "data" / "governance_augmentation_proposed.csv"
CATEGORY = "University Facts & Governance"
VARIANTS_PER_FACT = 4
DIVERSITY_THRESHOLD = 0.90  # reject a candidate whose cosine similarity to any existing/accepted phrasing exceeds this
MAX_ATTEMPTS_PER_FACT = VARIANTS_PER_FACT * 4

REPHRASE_PROMPT = (
    "Rephrase the following question into a genuinely different English "
    "phrasing -- different sentence structure and wording, same meaning, "
    "same specific facts/entities unchanged. This is for a university FAQ "
    "chatbot's training data, so it should sound like a real student "
    "question, not a formal rewrite. Output ONLY the rephrased question, "
    "nothing else, no quotes, no explanation.\n\nOriginal question: {question}"
)


def rephrase_diverse(question: str, seed: int) -> str:
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": REPHRASE_PROMPT.format(question=question),
            "stream": False,
            "options": {"temperature": 0.7, "seed": seed, "num_ctx": 512},
        },
        timeout=900,
    )
    return resp.json()["response"].strip().strip('"')


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT Question, Answer, Split FROM EnglishQA WHERE Category = ?", (CATEGORY,))
    rows = cur.fetchall()
    conn.close()

    facts = {}
    for q, a, split in rows:
        facts.setdefault(a, {"questions": [], "splits": []})
        facts[a]["questions"].append(q)
        facts[a]["splits"].append(split)

    print(f"{len(facts)} unique facts in {CATEGORY!r} (from {len(rows)} total rows)")
    val_covered = sum(1 for f in facts.values() if "val" in f["splits"])
    test_covered = sum(1 for f in facts.values() if "test" in f["splits"])
    print(f"  Currently covered in val: {val_covered}/{len(facts)}; in test: {test_covered}/{len(facts)}")

    embed_fn = Chroma1xEmbeddingFunction()

    records = []
    for fact_idx, (answer, info) in enumerate(facts.items()):
        source_q = info["questions"][0]
        existing_texts = list(info["questions"])
        existing_embs = np.array(embed_fn(existing_texts))

        accepted = []
        attempts = 0
        while len(accepted) < VARIANTS_PER_FACT and attempts < MAX_ATTEMPTS_PER_FACT:
            attempts += 1
            seed = abs(hash((answer, attempts))) % 100000
            candidate = rephrase_diverse(source_q, seed=seed)
            cand_emb = np.array(embed_fn([candidate]))[0]

            compare_pool = existing_embs if not accepted else np.vstack(
                [existing_embs] + [np.array(embed_fn([c]))[0:1] for c in accepted])
            norms = np.linalg.norm(compare_pool, axis=1) * np.linalg.norm(cand_emb)
            sims = (compare_pool @ cand_emb) / np.where(norms == 0, 1, norms)
            if sims.max() >= DIVERSITY_THRESHOLD:
                continue  # too similar to something already accepted/existing -- reject, don't count toward target
            accepted.append(candidate)

        # Assign new rows to whichever split is currently missing this fact,
        # prioritizing val first (the smallest, most fragile split for this
        # category), then test, then train -- the point is to close the
        # coverage gap, not just add more train rows to an already-larger split.
        target_splits = []
        if "val" not in info["splits"]:
            target_splits.append("val")
        if "test" not in info["splits"]:
            target_splits.append("test")
        while len(target_splits) < len(accepted):
            target_splits.append("train")

        for i, candidate in enumerate(accepted):
            records.append({
                "query_id": f"GOV-AUG-{fact_idx:03d}-{i}",
                "question": candidate,
                "answer": answer,
                "category": CATEGORY,
                "proposed_split": target_splits[i] if i < len(target_splits) else "train",
                "source_question": source_q,
            })
        print(f"  fact {fact_idx}: accepted {len(accepted)}/{VARIANTS_PER_FACT} "
              f"(attempts={attempts}), target_splits={target_splits[:len(accepted)]}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} proposed rows to {OUT_PATH}")
    print(f"Proposed val coverage after merge: "
          f"{val_covered + (out['proposed_split'] == 'val').nunique()}/{len(facts)} facts would have >=1 val row "
          f"(exact count depends on which facts got a 'val' target above)")
    print("THIS FILE IS PROPOSED ONLY -- knowledge_base.db and EnglishQA are unchanged. "
          "Review before merging; these are LLM-paraphrased questions over already-verified "
          "facts, not new facts, but naturalness has not been human-validated.")


if __name__ == "__main__":
    main()
