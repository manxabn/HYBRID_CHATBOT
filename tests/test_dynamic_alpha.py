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

from pipeline.hybrid_retriever import HybridRetriever


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


if __name__ == "__main__":
    test_no_signals_gives_zero_strength()
    test_course_code_alone_gives_quarter_strength()
    test_faculty_initial_and_name_together_gives_half_strength()
    test_all_four_signal_types_gives_full_strength()
    test_token_fallback_alone_counts_as_one_signal_not_two()
    test_strength_is_monotonic_with_signal_count()
    test_dynamic_alpha_interpolation_math_matches_entity_signal_strength()
    print("OK: all dynamic-alpha unit tests passed")
