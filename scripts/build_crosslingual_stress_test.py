"""
Builds the cross-lingual stress test identified as a gap in paper.tex
Section 4.7.3: query translation and BM25 code-mixed normalization showed
null effects on the main Banglish test set because that set is sampled from
BanglishQA's own stored questions, which already match the corpus's own
indexed spelling/phrasing by construction -- there is no genuine cross-
lingual divergence for either technique to bridge. This script instead
finds EnglishQA test-split content that has NO matching BanglishQA entry at
all (compared by exact AnswerEnglish/Answer text, case-insensitive), so a
Banglish-phrased question about it structurally CANNOT be answered by
BanglishQA-indexed content -- only a genuinely cross-lingual retrieval path
(the corpus's EnglishQA-indexed content, reached despite a Banglish query)
can succeed. 21 such EnglishQA test rows exist; deduplicated by unique
answer text (several are paraphrase duplicates of the same underlying fact,
identifiable via the same convention used elsewhere in this project), this
yields a small but structurally sound stress test.

Each English question is rephrased into natural Banglish by the same local
LLM used for generation (Llama-3.1-8B via Ollama) -- disclosed here as the
construction method, not presented as human-collected data. This mirrors
standard code-switched-NLP dataset construction practice (LLM/MT-assisted
rephrasing, e.g. GLUECoS-style benchmarks) rather than inventing a novel
method; the point of this script is the STRUCTURAL selection criterion
above (no-BanglishQA-match), not the rephrasing step itself.

Usage: python scripts/build_crosslingual_stress_test.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry
import requests

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "data" / "test_queries_crosslingual_stress.csv"

BANGLISH_REPHRASE_PROMPT = (
    "Rephrase the following English question into natural Banglish -- "
    "Bengali-English code-mixed text written in Latin script, the way a "
    "BRAC University student would casually type it in a chat message. Keep "
    "the same meaning and the same specific entities (course codes, names, "
    "numbers) unchanged. Output ONLY the rephrased Banglish question, "
    "nothing else, no quotes, no explanation.\n\nEnglish question: {question}"
)


def rephrase_to_banglish(question: str) -> str:
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": BANGLISH_REPHRASE_PROMPT.format(question=question),
            "stream": False,
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 512},
        },
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
        "WHERE Split='test' AND Category != 'Out of Scope / Unanswerable'"
    )
    rows = cur.fetchall()
    conn.close()

    english_only = [r for r in rows if r[2] and r[2].strip().lower() not in banglish_answers]
    print(f"English test rows with no matching BanglishQA answer: {len(english_only)}/{len(rows)}")

    seen_answers = set()
    deduped = []
    for row_id, question, answer, category in english_only:
        key = answer.strip().lower()
        if key in seen_answers:
            continue
        seen_answers.add(key)
        deduped.append((row_id, question, answer, category))
    print(f"Deduplicated by unique answer text: {len(deduped)} rows")

    records = []
    for i, (row_id, question, answer, category) in enumerate(deduped):
        banglish_query = rephrase_to_banglish(question)
        records.append({
            "query_id": f"XL{i+1:03d}",
            "query": banglish_query,
            "reference_answer": answer,
            "source_english_question": question,
            "category": category,
        })
        print(f"  [{i+1}/{len(deduped)}] {question[:60]!r} -> {banglish_query[:60]!r}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
