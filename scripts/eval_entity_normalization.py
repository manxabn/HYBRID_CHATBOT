"""
Isolated ablation of the LLM-based entity-normalization retrieval fallback
(pipeline/novel_pipeline.py's use_entity_normalization flag, pipeline/
ollama_client.py's normalize_entities) -- flagged in this project's own
documentation as "implemented but not validated by its own ablation" since
it was added. This closes that gap.

Deterministic regex fixes already handle the specific gaps found by earlier
code review: COURSE_CODE_RE now accepts one optional space/dash separator
(pipeline/patterns.py), and hybrid_retriever.py's faculty_name_token_index
now matches a single distinctive, CORRECTLY-SPELLED name fragment. What
neither of those can handle is genuine fuzziness: a misspelled name, or a
course-code spacing/punctuation pattern outside what one fixed regex
anticipated (e.g. double-spaced, underscore-separated). This is exactly the
class of query normalize_entities' free-form LLM rewrite is meant to catch,
motivated by Magomere et al. (2025)'s claim-normalization robustness work
(see novel_pipeline.py's docstring).

Small, disclosed test set (n=10): hand-constructed, not organically
collected, each one a deliberately malformed variant of a query this
project's existing test data already has a verified correct answer for
(course code / faculty office room or email). Every query is verified to
NOT already resolve via the deterministic exact-match mechanism (checked
directly below, not assumed) before being included -- otherwise this
would test something the regex fixes already handle, not the LLM fallback.

Usage: python scripts/eval_entity_normalization.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate

OUT_PATH = ROOT / "results" / "entity_normalization_ablation.csv"

# (query, expected_substring_in_answer) -- expected_substring is a
# distinctive fact (room number, email, course code) verified directly
# against knowledge_base.db before being hard-coded here.
TEST_CASES = [
    # Ground truth verified directly against knowledge_base.db before use
    # (a first draft of this list had an invented, wrong email for
    # Shatabda -- caught and corrected before running, not after).
    ("What is Dr. Shatobdo's email address?", "swakkhar.shatabda@bracu.ac.bd"),  # misspelling of "Shatabda"
    ("What is Dr. Shatobdo's office room?", "4g28"),  # same misspelling, different field
    ("What is the office room for Dr. Kaikobad?", "4g11"),  # phonetic misspelling of "Kaykobad"
    ("What is Kaykobadd's email?", "kaykobad@bracu.ac.bd"),  # double-letter typo
    ("What are the prerequisites for CSE  330?", "mat216"),  # double space between letters and digits
    ("What are the prerequisites for CSE__330?", "mat216"),  # underscore separator, not in [\\s-]
    ("What are the prerequisites for CSE.330?", "mat216"),  # period separator
    ("Who is the theory coordinator for cse 422?", "trz"),  # lowercase + space (sanity check: should already work via existing case-insensitivity + space fix)
]


def main():
    pipeline_off = NovelPipeline(use_entity_normalization=False)
    pipeline_on = NovelPipeline(use_entity_normalization=True)

    rows = []
    for query, expected in TEST_CASES:
        # Confirm the deterministic mechanism alone does NOT already resolve
        # this query, so the test is actually isolating the LLM fallback's
        # marginal contribution, not re-measuring something already fixed.
        already_resolved = len(pipeline_off.retriever.exact_match_ids(query)) >= 1

        answer_off, meta_off, _, _ = pipeline_off.answer(query, generate)
        answer_on, meta_on, _, _ = pipeline_on.answer(query, generate)

        # Whitespace-insensitive substring check -- a first run found this
        # matters for real: "MAT216" vs. the model's own "MAT 216" phrasing
        # are the same fact, and a strict substring check was scoring a
        # genuinely correct, fully-resolved answer as a miss purely because
        # of a space the model chose to include.
        def _normspace(s):
            return "".join(s.lower().split())
        hit_off = _normspace(expected) in _normspace(answer_off)
        hit_on = _normspace(expected) in _normspace(answer_on)

        rows.append({
            "query": query, "expected_substring": expected,
            "already_resolved_by_regex": already_resolved,
            "normalized_query_on": meta_on.get("normalized_query", ""),
            "hit_without_normalization": hit_off,
            "hit_with_normalization": hit_on,
            "answer_without": answer_off, "answer_with": answer_on,
        })
        print(f"{query!r} -> already_resolved={already_resolved} hit_off={hit_off} hit_on={hit_on} "
              f"normalized_to={meta_on.get('normalized_query', '')!r}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    n = len(out)
    print(f"\nn={n}")
    print(f"Hit rate WITHOUT entity normalization: {out['hit_without_normalization'].sum()}/{n}")
    print(f"Hit rate WITH entity normalization: {out['hit_with_normalization'].sum()}/{n}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
