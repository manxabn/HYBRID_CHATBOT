"""
Expands the cross-lingual stress test (scripts/build_crosslingual_stress_
test.py, 9 queries) to more items WITHOUT needing new source content: the
structural bottleneck is genuinely EnglishQA-only facts (only 9 unique
ones exist in the test split), not phrasing diversity, so we generate
MULTIPLE distinct Banglish phrasings of each of the same 9 underlying
facts rather than inventing new facts. This follows the same LLM-assisted
rephrasing methodology as the original script, disclosed the same way,
with one deliberate deviation: temperature=0.7 (not the project's
otherwise-universal temperature=0.0) specifically to obtain genuinely
different phrasings across repeated calls on the same input -- greedy
decoding would just return the same rephrasing every time. This trades the
project's usual reproducibility-by-determinism for phrasing diversity,
which is exactly what is needed here; every other retrieval-time or
generation-time LLM call in this project remains deterministic.

Doğruöz, Sitaram & Yong (2023, Findings of EMNLP, arXiv:2310.20470) caution
that LLM-generated code-switched text can look more "textbook" than
naturally-occurring code-switching, and that datasets built through
elicitation or generation often don't reflect how real bilingual speakers
actually mix languages. We do not have a second human rater available to
validate naturalness of the expanded variants (the same standing
limitation as the rest of this project's human-annotation gap) -- this is
disclosed directly in the output and must be disclosed in any paper text
citing the expanded set's results, not silently treated as equivalent to
the original, more conservative 9-query set.

Usage: python scripts/expand_crosslingual_stress_test.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry

IN_PATH = ROOT / "data" / "test_queries_crosslingual_stress.csv"
OUT_PATH = ROOT / "data" / "test_queries_crosslingual_stress_expanded.csv"
VARIANTS_PER_QUESTION = 3

BANGLISH_REPHRASE_PROMPT = (
    "Rephrase the following English question into natural Banglish -- "
    "Bengali-English code-mixed text written in Latin script, the way a "
    "BRAC University student would casually type it in a chat message. Keep "
    "the same meaning and the same specific entities (course codes, names, "
    "numbers) unchanged. Vary your wording and sentence structure -- do not "
    "just produce a literal word-for-word transliteration. Output ONLY the "
    "rephrased Banglish question, nothing else, no quotes, no "
    "explanation.\n\nEnglish question: {question}"
)


def rephrase_to_banglish_diverse(question: str, seed: int) -> str:
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": MODEL,
            "prompt": BANGLISH_REPHRASE_PROMPT.format(question=question),
            "stream": False,
            # Deliberate deviation from this project's universal temperature=0.0
            # -- see module docstring: greedy decoding would return an
            # identical rephrasing on every call for the same input, which is
            # the opposite of what expanding phrasing diversity requires.
            "options": {"temperature": 0.7, "seed": seed, "num_ctx": 512},
        },
        timeout=900,
    )
    return resp.json()["response"].strip().strip('"')


def main():
    base = pd.read_csv(IN_PATH)
    print(f"Expanding {len(base)} base queries x {VARIANTS_PER_QUESTION} variants each...")

    records = []
    seen_texts_per_source = {}
    for _, row in base.iterrows():
        source_q = row["source_english_question"]
        seen = seen_texts_per_source.setdefault(row["query_id"], set())
        variant_idx = 0
        attempts = 0
        while variant_idx < VARIANTS_PER_QUESTION and attempts < VARIANTS_PER_QUESTION * 3:
            attempts += 1
            seed = 1000 * (attempts) + hash(row["query_id"]) % 1000
            variant = rephrase_to_banglish_diverse(source_q, seed=abs(seed) % 100000)
            if variant.strip().lower() in seen:
                continue  # got a duplicate phrasing, retry for genuine diversity
            seen.add(variant.strip().lower())
            variant_idx += 1
            records.append({
                "query_id": f"{row['query_id']}-v{variant_idx}",
                "query": variant,
                "reference_answer": row["reference_answer"],
                "source_english_question": source_q,
                "category": row["category"],
            })
            print(f"  {row['query_id']}-v{variant_idx}: {variant[:70]!r}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} expanded rows (from {len(base)} base queries) to {OUT_PATH}")
    print("NOTE: naturalness of these LLM-generated variants has not been human-validated "
          "(Doğruöz, Sitaram & Yong 2023's representativeness caveat) -- disclose this "
          "alongside any result computed on this expanded set.")


if __name__ == "__main__":
    main()
