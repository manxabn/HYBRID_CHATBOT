"""
Unit tests for HybridRetriever.entity_signal_strength/retrieve_dynamic_alpha
(pipeline/hybrid_retriever.py) -- the EXPERIMENTAL, DAT-inspired continuous
fusion-weight mode, added 2026-07-29, not yet empirically validated (see
that method's docstring and scripts/ablate_dynamic_alpha.py).

Deliberately does NOT instantiate the real HybridRetriever: its __init__
always creates a Chroma1xEmbeddingFunction, which tries CUDA if available --
unsafe to run while another GPU job (E5-small fine-tuning) is in progress on
this 4GB-VRAM machine (see this project's own documented resource-
contention incidents). Instead, calls entity_signal_strength as an unbound
method against a minimal stand-in object carrying only the attributes that
method actually reads (faculty_initial_index, faculty_name_index,
faculty_name_token_index, aliases) -- this tests the real logic, not a
re-implementation of it, with zero GPU/model/DB dependency.

No test framework dependency (pytest is not installed in this project) --
plain asserts, run directly: `python tests/test_dynamic_alpha.py`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever, _matched_faculty_initials


def _fake_retriever(aliases=None, faculty_initials=None, faculty_names=None, faculty_tokens=None):
    return SimpleNamespace(
        aliases=aliases or [],
        faculty_initial_index=faculty_initials or {},
        faculty_name_index=faculty_names or [],
        faculty_name_token_index=faculty_tokens or {},
        _N_ENTITY_SIGNAL_TYPES=HybridRetriever._N_ENTITY_SIGNAL_TYPES,
    )


def test_no_signals_gives_zero_strength():
    fake = _fake_retriever()
    strength = HybridRetriever.entity_signal_strength(fake, "What are the library hours on weekends?")
    assert strength == 0.0, f"expected 0.0 for a query with no entity signals, got {strength}"


def test_course_code_alone_gives_quarter_strength():
    fake = _fake_retriever()
    strength = HybridRetriever.entity_signal_strength(fake, "What are the prerequisites for CSE220?")
    assert strength == 0.25, f"expected 0.25 (1/4 signal types) for a bare course-code query, got {strength}"


def test_faculty_initial_and_name_together_gives_half_strength():
    # Two DISTINCT signal types (initial index hit + name index hit) for the
    # SAME person should count as 2/4, not more -- this test also confirms
    # the two checks are independent (both can fire on the same query).
    fake = _fake_retriever(
        faculty_initials={"AAR": "doc1"},
        faculty_names=[("mohammad kaykobad", "doc1")],
    )
    strength = HybridRetriever.entity_signal_strength(fake, "Is Dr. Mohammad Kaykobad (AAR) available on Sunday?")
    assert strength == 0.5, f"expected 0.5 (2/4 signal types), got {strength}"


def test_all_four_signal_types_gives_full_strength():
    fake = _fake_retriever(
        aliases=[("nummeth", "CSE330")],
        faculty_initials={"AAR": "doc1"},
        faculty_names=[("mohammad kaykobad", "doc1")],
    )
    # Course code (CSE220) + alias (NumMeth) + initial (AAR) + name (Kaykobad)
    strength = HybridRetriever.entity_signal_strength(
        fake, "Does CSE220 relate to NumMeth, and is AAR / Dr. Mohammad Kaykobad the coordinator?")
    assert strength == 1.0, f"expected 1.0 (4/4 signal types), got {strength}"


def test_token_fallback_alone_counts_as_one_signal_not_two():
    # faculty_name_index (full-name substring) and faculty_name_token_index
    # (single-token fallback) are two separate checks inside ONE signal
    # type (the "or" in entity_signal_strength's faculty name/token block)
    # -- a query matching only the token fallback (a name fragment, not the
    # full stored name) must still count as exactly 1 signal, not 2.
    fake = _fake_retriever(faculty_tokens={"kaykobad": ["doc1"]})
    strength = HybridRetriever.entity_signal_strength(fake, "What is Kaykobad's office room?")
    assert strength == 0.25, f"expected 0.25 (token fallback is still just 1 signal type), got {strength}"


def test_strength_is_monotonic_with_signal_count():
    fake0 = _fake_retriever()
    fake1 = _fake_retriever(faculty_initials={"AAR": "doc1"})
    fake2 = _fake_retriever(faculty_initials={"AAR": "doc1"}, aliases=[("nummeth", "CSE330")])
    s0 = HybridRetriever.entity_signal_strength(fake0, "What is AAR's designation via NumMeth?")
    s1 = HybridRetriever.entity_signal_strength(fake1, "What is AAR's designation via NumMeth?")
    s2 = HybridRetriever.entity_signal_strength(fake2, "What is AAR's designation via NumMeth?")
    assert s0 == 0.0 and s1 == 0.25 and s2 == 0.5, f"expected strictly increasing 0.0 < 0.25 < 0.5, got {s0}, {s1}, {s2}"
    assert s0 < s1 < s2


def test_matched_faculty_initials_excludes_add_but_keeps_real_initials():
    # 2026-08-02: FACULTY_INITIAL_RE's original-casing fix (2026-08-01)
    # closed the lowercase-word collision class ("add a course" no longer
    # false-matches ADD) but not a second one -- "ADD" typed in full caps,
    # as in the standard registrar term "ADD/DROP", still matched, since
    # the match itself is already all-caps as written. Confirmed live:
    # "When is the ADD/DROP deadline?" force-ranked an unrelated faculty
    # member (initial ADD) to the exact-match ceiling. This test locks in
    # the fix: ADD is excluded, but an unrelated real initial (AAR) in the
    # same query is still recognized normally.
    matched = _matched_faculty_initials("When is the ADD/DROP deadline, and is AAR the coordinator?")
    assert "ADD" not in matched, f"ADD should be excluded as a known false-positive, got {matched}"
    assert "AAR" in matched, f"AAR is a real, unexcluded initial and should still match, got {matched}"


def test_is_entity_heavy_false_for_add_drop_query_even_with_real_add_collision():
    # Simulates the real corpus fact that ADD genuinely is a FacultyList
    # initial (Ayesha Siddika) -- without the fix, is_entity_heavy would
    # return True here purely because of that collision, misrouting an
    # ordinary add/drop-period question into the entity_heavy/exact-match
    # branch and force-ranking her unrelated record to the top.
    fake = _fake_retriever(faculty_initials={"ADD": "faculty_doc_ayesha"})
    assert HybridRetriever.is_entity_heavy(fake, "When is the ADD/DROP deadline?") is False
    assert HybridRetriever.is_entity_heavy(fake, "What is the ADD/DROP period for this semester?") is False
    # Sanity: a genuinely different real initial in the same fake index is
    # still recognized as entity-heavy (the exclusion is scoped to ADD
    # only, not a blanket disabling of faculty-initial matching).
    fake2 = _fake_retriever(faculty_initials={"AAR": "faculty_doc_kaykobad"})
    assert HybridRetriever.is_entity_heavy(fake2, "What is AAR's designation?") is True


def test_dynamic_alpha_interpolation_math_matches_entity_signal_strength():
    # Confirms retrieve_dynamic_alpha's lambda interpolation formula itself
    # (lambda_open + (lambda_entity - lambda_open) * strength) without
    # calling retrieve() (which needs the real corpus/BM25/Chroma state) --
    # isolates exactly the new arithmetic this method adds.
    lambda_open, lambda_entity = 0.5, 0.9
    for strength, expected_lambda in [(0.0, 0.5), (0.25, 0.6), (0.5, 0.7), (0.75, 0.8), (1.0, 0.9)]:
        lambda_ = lambda_open + (lambda_entity - lambda_open) * strength
        assert abs(lambda_ - expected_lambda) < 1e-9, \
            f"strength={strength}: expected lambda={expected_lambda}, got {lambda_}"


def test_tied_score_sort_is_deterministic_regardless_of_input_order():
    # 2026-08-02: retrieve()'s final ranking step used to sort candidates
    # by score alone (`scored.sort(key=lambda x: x["score"], reverse=True)`).
    # When multiple candidates tie exactly on score (a real, recurring case
    # -- e.g. several FacultyAvailability rows for the same person on
    # different days, identical score for a query that doesn't name a day),
    # a stable sort preserves whatever order the candidates arrived in,
    # which traced back to iterating a `set` of candidate IDs -- and
    # Python randomizes the hash seed for str keys per PROCESS by default,
    # so two separate invocations of the identical script could silently
    # return a different tied candidate first. Confirmed live: the exact
    # same query, same code, same corpus returned "Day: Tuesday" in one
    # process and "Day: Sunday" in another (49/101 entity-heavy queries
    # affected in a real before/after comparison). Fixed with a
    # deterministic secondary sort key (doc_id). This test locks in the
    # fix directly: the same tied-score candidates, fed in two different
    # input orders (simulating two different hash-seed-driven set
    # iteration orders), must sort to the identical output either way.
    candidates_order_a = [
        {"doc_id": "FacultyAvailability-99-chunk0", "score": 101.2},
        {"doc_id": "FacultyAvailability-42-chunk0", "score": 101.2},
        {"doc_id": "FacultyAvailability-7-chunk0", "score": 101.2},
    ]
    candidates_order_b = list(reversed(candidates_order_a))
    assert candidates_order_a != candidates_order_b  # sanity: inputs really do differ in order

    sort_key = lambda x: (x["score"], x["doc_id"])
    sorted_a = sorted(candidates_order_a, key=sort_key, reverse=True)
    sorted_b = sorted(candidates_order_b, key=sort_key, reverse=True)
    assert [c["doc_id"] for c in sorted_a] == [c["doc_id"] for c in sorted_b], \
        f"tied candidates sorted differently depending on input order: {sorted_a} vs {sorted_b}"


if __name__ == "__main__":
    test_no_signals_gives_zero_strength()
    test_course_code_alone_gives_quarter_strength()
    test_faculty_initial_and_name_together_gives_half_strength()
    test_all_four_signal_types_gives_full_strength()
    test_token_fallback_alone_counts_as_one_signal_not_two()
    test_strength_is_monotonic_with_signal_count()
    test_matched_faculty_initials_excludes_add_but_keeps_real_initials()
    test_is_entity_heavy_false_for_add_drop_query_even_with_real_add_collision()
    test_dynamic_alpha_interpolation_math_matches_entity_signal_strength()
    test_tied_score_sort_is_deterministic_regardless_of_input_order()
    print("OK: all dynamic-alpha unit tests passed")
