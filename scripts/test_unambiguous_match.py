"""
Isolated retrieval-only test of UNAMBIGUOUS_MATCH_SCORE (hybrid_retriever.py):
the score-ceiling guarantee added 2026-07-27 that forces a SINGLE unambiguous
exact-match candidate to top-1, on top of the already-validated EXACT_MATCH_BONUS
additive bonus. This constant has been live in every round since (round K
baselines, reranker ablation) but was never isolated-tested on its own --
every other round has it either fully on or the code didn't exist yet, so
there's no direct evidence of what it changes by itself.

Method: retrieval-only (no generation needed, cheap), on the 100 entity-heavy
queries in data/test_queries.csv (English) via retrieve_adaptive (the actual
novel-pipeline routing path, which is where exact-match/RRF fusion is used).
Compare top-1 accuracy (is the correct row ranked FIRST, not just in top-5)
with UNAMBIGUOUS_MATCH_SCORE at its live value (100.0) vs. patched to 0.0
(EXACT_MATCH_BONUS=0.3 still active in both conditions) -- isolating exactly
the marginal contribution of the score-ceiling on top of the additive bonus.

Top-1 (not top-5) is the right metric here because the docstring's own claim
is specifically about guaranteeing RANK, not presence in the candidate set --
EXACT_MATCH_BONUS alone was already shown (2026-07-27 debugging session) to
usually be enough for top-5 presence; the open question is whether it's
enough for top-1 in every case or only most.

Correct-row matching: unlike the Banglish test set (where reference_answer
is a verbatim corpus AnswerEnglish field, so substring match works directly,
see lambda_sweep_banglish.py), this English entity-heavy set's reference
answers are COMPOSED sentences (paper.tex Section 4.2: "reference answer was
composed directly from the relevant structured fields"), e.g. "The
prerequisite for CSE423 is MAT216 (HP)." is not a substring of the corpus
chunk "Course: CSE423\nPrerequisite: MAT216 (HP)\n...". Confirmed live by
inspecting a real retrieval: substring match against the reference answer
gave 0/100 for BOTH conditions, which is a broken test, not a real result.
Instead: extract every course-code token (COURSE_CODE_RE) present in the
reference answer, and require ALL of them to appear in the candidate's text
-- this correctly distinguishes "Course: CSE423\nPrerequisite: MAT216 (HP)"
(contains both CSE423 and MAT216) from a same-course but wrong-table
candidate like "Course: CSE423-03\n..." (contains CSE423 but not MAT216).
If the reference answer itself contains no course code (e.g. a coordinator
or faculty name answer), fall back to requiring the query's own extracted
code(s) to match the candidate's canonical Course metadata field. For
faculty queries (2026-07-28 extension, see hybrid_retriever.py's
faculty_initial_index/faculty_name_index), reference answers contain an
email address -- the single most discriminative token available (unlike a
room number, which multiple faculty can share) -- so email tokens are
checked the same way course codes are: extracted from the reference answer
via EMAIL_RE and required to all appear in the candidate's text.

Usage: python scripts/test_unambiguous_match.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from pipeline.hybrid_retriever import COURSE_CODE_RE, FACULTY_INITIAL_RE, _normalize_name

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _is_correct(candidate, query: str, reference_answer: str) -> bool:
    ref_codes = {m.upper() for m in COURSE_CODE_RE.findall(reference_answer)}
    ref_emails = {m.lower() for m in EMAIL_RE.findall(reference_answer)}
    text_upper = candidate["text"].upper()
    text_lower = candidate["text"].lower()
    if ref_codes:
        return all(code in text_upper for code in ref_codes)
    if ref_emails:
        return all(email in text_lower for email in ref_emails)
    query_codes = {m.upper() for m in COURSE_CODE_RE.findall(query)}
    if query_codes:
        cand_course = str(candidate["metadata"].get("Course", "")).upper()
        cand_canonical = COURSE_CODE_RE.match(cand_course)
        cand_canonical = cand_canonical.group(0) if cand_canonical else ""
        return cand_canonical in query_codes
    # No course code or email in play at all -- a faculty room/designation
    # query. Fall back to the same identity signals hybrid_retriever.py's
    # own faculty exact-match uses: does the candidate's Initial appear as a
    # standalone token in the query, or does the candidate's Name appear
    # (normalized) within the normalized query.
    cand_initial = str(candidate["metadata"].get("Initial", "")).upper()
    if cand_initial and any(m.group(0) == cand_initial for m in FACULTY_INITIAL_RE.finditer(query.upper())):
        return True
    cand_name = _normalize_name(str(candidate["metadata"].get("Name", "")))
    norm_query = _normalize_name(query)
    return bool(cand_name) and cand_name in norm_query


QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "unambiguous_match_test.csv"


def run(retriever, df, label):
    n_top1, n_top5, n = 0, 0, 0
    rows = []
    for _, r in df.iterrows():
        results, _ = retriever.retrieve_adaptive(r["query"])
        hit_ranks = [i for i, c in enumerate(results) if _is_correct(c, r["query"], str(r["reference_answer"]))]
        top1 = bool(hit_ranks) and hit_ranks[0] == 0
        top5 = bool(hit_ranks)
        n_top1 += top1
        n_top5 += top5
        n += 1
        rows.append({"query_id": r["query_id"], "condition": label, "top1_hit": top1, "top5_hit": top5})
    print(f"{label}: top1_accuracy={n_top1/n:.3f} top5_accuracy={n_top5/n:.3f} (n={n})")
    return rows, n_top1 / n, n_top5 / n


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = df[df["is_entity_heavy"] == True].reset_index(drop=True)
    print(f"Entity-heavy queries: {len(df)}")

    all_rows = []

    # Condition 1: live value (100.0) -- current production behavior.
    retriever = hr.HybridRetriever()
    rows_on, top1_on, top5_on = run(retriever, df, "unambiguous_on")
    all_rows.extend(rows_on)

    # Condition 2: patched to 0.0 -- EXACT_MATCH_BONUS (0.3) still active,
    # isolating just the ceiling's marginal contribution. Same retriever
    # instance/index reused; only the module-level constant referenced
    # inside _score_linear/_score_rrf at call time changes.
    hr.UNAMBIGUOUS_MATCH_SCORE = 0.0
    rows_off, top1_off, top5_off = run(retriever, df, "unambiguous_off")
    all_rows.extend(rows_off)

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"\nDelta (on - off): top1={top1_on - top1_off:+.3f}  top5={top5_on - top5_off:+.3f}")


if __name__ == "__main__":
    main()
