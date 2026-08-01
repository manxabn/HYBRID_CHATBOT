"""
Extends the cross-lingual stress test (build_crosslingual_stress_test.py)
with the val-split's own EnglishQA-only facts, not just test-split. The
original script used only Split='test' (21 rows, 9 unique facts after
dedup); Split='train' rows are correctly excluded from any stress-test
expansion since the deployed embedding model was fine-tuned directly on
train-split (question, answer) pairs -- using them would leak training
data into what is supposed to be a held-out evaluation. Split='val' rows
were never trained on either (same status as test for this purpose, this
project's own convention elsewhere, e.g. eval_embeddings_held_out.py),
so they are a legitimate, honest expansion: 8 val-split EnglishQA-only
rows exist, deduplicating to 4 unique additional facts.

Same Banglish-rephrasing methodology as the original script (identical
prompt, temperature=0.0, seed=42) -- reused, not reinvented -- applied
only to these 4 new facts, then combined with the original 9 into one
combined n=13 stress-test file.

Usage: python scripts/build_crosslingual_stress_test_valsplit.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import OLLAMA_URL, MODEL, post_with_retry
from build_crosslingual_stress_test import BANGLISH_REPHRASE_PROMPT

DB_PATH = ROOT / "knowledge_base.db"
ORIGINAL_PATH = ROOT / "data" / "test_queries_crosslingual_stress.csv"
OUT_PATH = ROOT / "data" / "test_queries_crosslingual_stress_valexpanded.csv"


def rephrase_to_banglish(question: str) -> str:
    resp = post_with_retry(
        OLLAMA_URL,
        {"model": MODEL, "prompt": BANGLISH_REPHRASE_PROMPT.format(question=question),
         "stream": False, "options": {"temperature": 0.0, "seed": 42, "num_ctx": 512}},
        timeout=900,
    )
    return resp.json()["response"].strip().strip('"')


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT AnswerEnglish FROM BanglishQA")
    banglish_answers = {r[0].strip().lower() for r in cur.fetchall() if r[0]}
    cur.execute(
        "SELECT id, Question, Answer, Category FROM EnglishQA "
        "WHERE Split='val' AND Category != 'Out of Scope / Unanswerable'"
    )
    rows = cur.fetchall()
    conn.close()

    english_only = [r for r in rows if r[2] and r[2].strip().lower() not in banglish_answers]
    print(f"Val-split rows with no matching BanglishQA answer: {len(english_only)}/{len(rows)}")

    seen_answers = set()
    deduped = []
    for row_id, question, answer, category in english_only:
        key = answer.strip().lower()
        if key in seen_answers:
            continue
        seen_answers.add(key)
        deduped.append((row_id, question, answer, category))
    print(f"Deduplicated: {len(deduped)} unique new facts")

    original = pd.read_csv(ORIGINAL_PATH)
    start_i = len(original) + 1

    records = []
    for i, (row_id, question, answer, category) in enumerate(deduped):
        banglish_query = rephrase_to_banglish(question)
        records.append({
            "query_id": f"XL{start_i+i:03d}",
            "query": banglish_query,
            "reference_answer": answer,
            "source_english_question": question,
            "category": category,
        })
        print(f"  [{i+1}/{len(deduped)}] {question[:60]!r} -> {banglish_query[:60]!r}", flush=True)

    new_df = pd.DataFrame(records)
    combined = pd.concat([original, new_df], ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined)} total rows ({len(original)} original + {len(new_df)} new) to {OUT_PATH}")


if __name__ == "__main__":
    main()
