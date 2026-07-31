"""
Regression test for pipeline/patterns.py -- specifically added to catch
the exact drift that already happened once: pipeline/prerequisite_graph.py
used to define its own COURSE_CODE_RE that lacked the letter-suffix fix
(CSE490B vs. CSE490C) present in pipeline/hybrid_retriever.py's copy. Both
now import the single definition in patterns.py, so this test also
guarantees there is only one place left that could regress.

No test framework dependency (pytest is not installed in this project) --
plain asserts, run directly: `python tests/test_patterns.py`. Exits 0 and
prints "OK" on success, raises AssertionError with a clear message and
non-zero exit on failure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.patterns import COURSE_CODE_RE, FULL_COURSE_ID_RE, canonicalize_course_code
import pipeline.hybrid_retriever as hybrid_retriever
import pipeline.prerequisite_graph as prerequisite_graph


def test_letter_suffixed_course_codes_match():
    for code in ["CSE490B", "CSE490C", "CSE111", "MAT216"]:
        assert COURSE_CODE_RE.fullmatch(code), f"COURSE_CODE_RE should fully match {code!r}"


def test_letter_suffixed_course_codes_distinguished_in_context():
    # The original bug: without the optional trailing letter, both CSE490B
    # and CSE490C collapsed into the same "CSE490" match, conflating two
    # distinct courses. Confirm search() on a real query-shaped string
    # extracts the FULL code, not the truncated base.
    m = COURSE_CODE_RE.search("Who is the theory coordinator for CSE490B?".upper())
    assert m and m.group(0) == "CSE490B", f"expected 'CSE490B', got {m.group(0) if m else None!r}"
    m = COURSE_CODE_RE.search("Who is the theory coordinator for CSE490C?".upper())
    assert m and m.group(0) == "CSE490C", f"expected 'CSE490C', got {m.group(0) if m else None!r}"


def test_full_course_id_re_captures_section_suffix():
    m = FULL_COURSE_ID_RE.fullmatch("CSE111-07")
    assert m and m.group(0) == "CSE111-07"
    m = FULL_COURSE_ID_RE.fullmatch("CSE260-05A")
    assert m and m.group(0) == "CSE260-05A"


def test_spaced_and_dashed_course_codes_now_match():
    # 2026-07-28 fix: "CSE 220" and "CSE-220" previously matched NOTHING at
    # all (confirmed live before this fix), which zeroed out exact-match
    # and, combined with the now-brittle entity-heavy abstention threshold,
    # would make the system always decline a perfectly answerable query
    # solely because of how the course code was spaced.
    for query in ["What are the prerequisites for CSE 220?", "What are the prerequisites for CSE-220?",
                  "What are the prerequisites for CSE220?"]:
        matches = COURSE_CODE_RE.findall(query.upper())
        assert matches, f"expected a match in {query!r}, got none"
        assert canonicalize_course_code(matches[0]) == "CSE220", \
            f"expected canonicalized 'CSE220' from {query!r}, got {canonicalize_course_code(matches[0])!r}"


def test_course_code_re_does_not_false_positive_on_dates_or_mid_word():
    # 2026-07-29 fix: found via code-review audit against the actual
    # test-query CSVs, not a hypothetical -- these exact rows exist in
    # test_queries.csv/roundA/roundB/roundC and were silently mis-flagged
    # as entity_heavy (routed through the heavily lexical-weighted RRF
    # branch) purely because a 4+-digit year's first 3 digits, or a
    # non-course word's trailing 2-4 letters, happened to look like a bare
    # course code.
    no_match_queries = [
        "Reason for late fee appearing on pay slip of scholarship recipients after 29 June 2025?",
        "Purpose of the Wishlist event for Summer 2025?",
        "The lab has 30 seats and floor 220 hallway access.",
    ]
    for query in no_match_queries:
        assert not COURSE_CODE_RE.search(query), \
            f"expected no course-code match in {query!r}, got {COURSE_CODE_RE.search(query).group(0)!r}"


def test_hybrid_retriever_and_prerequisite_graph_share_the_same_pattern_object():
    # The strongest possible regression guard: not just equal behavior, but
    # literally the same compiled regex object in memory, so it is
    # structurally impossible for the two modules' course-code recognition
    # to diverge again the way it did before this fix.
    assert hybrid_retriever.COURSE_CODE_RE is COURSE_CODE_RE
    assert prerequisite_graph.COURSE_CODE_RE is COURSE_CODE_RE


if __name__ == "__main__":
    test_letter_suffixed_course_codes_match()
    test_letter_suffixed_course_codes_distinguished_in_context()
    test_full_course_id_re_captures_section_suffix()
    test_spaced_and_dashed_course_codes_now_match()
    test_course_code_re_does_not_false_positive_on_dates_or_mid_word()
    test_hybrid_retriever_and_prerequisite_graph_share_the_same_pattern_object()
    print("OK: all pattern regression tests passed")
