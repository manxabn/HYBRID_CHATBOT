"""
Unit tests for pipeline/conformal_abstention.py (claim-level NLI back-off,
added this session, previously untested) and pipeline/novel_pipeline.py's
_zigzag_by_confidence (hand-verified during code review, but never given a
regression test of its own).

Deliberately avoids loading the real ~440MB cross-encoder/nli-MiniLM2-L6-H768
model, same rationale as tests/test_dynamic_alpha.py's avoidance of the real
HybridRetriever/Chroma embedding function: these tests monkeypatch
score_claims (the one function that actually calls the NLI model) with a
lookup-table stand-in, so the tests exercise this module's own selection/
routing/threshold logic, not sentence-transformers or torch.

No test framework dependency assumed beyond what's already used elsewhere in
this project (pytest, confirmed installed and passing for tests/test_
patterns.py) -- plain asserts, also runnable directly:
`python tests/test_conformal_abstention.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.conformal_abstention as ca
from pipeline.novel_pipeline import _zigzag_by_confidence


# ---------------------------------------------------------------------------
# backoff_filter: entity_heavy route (no NLI model involved at all -- this
# route defers entirely to exact_match_any, see module docstring)
# ---------------------------------------------------------------------------

def test_entity_heavy_route_with_exact_match_retains_fully():
    result = ca.backoff_filter("Dr. Kaykobad's office is 4G11.", context="irrelevant",
                                route="entity_heavy", exact_match_any=True)
    assert result["retained_fraction"] == 1.0
    assert result["filtered_answer"] == "Dr. Kaykobad's office is 4G11."
    assert result["claim_scores"] == []


def test_entity_heavy_route_without_exact_match_retains_nothing():
    result = ca.backoff_filter("Dr. Kaykobad's office is 4G11.", context="irrelevant",
                                route="entity_heavy", exact_match_any=False)
    assert result["retained_fraction"] == 0.0
    assert result["filtered_answer"] == ""


# ---------------------------------------------------------------------------
# backoff_filter: open_ended route, no context to check against
# ---------------------------------------------------------------------------

def test_open_ended_route_with_empty_context_is_not_filtered():
    result = ca.backoff_filter("Some answer.", context="", route="open_ended")
    assert result["filtered_answer"] == "Some answer."
    assert result["retained_fraction"] == 0.0
    assert result["claim_scores"] == []


# ---------------------------------------------------------------------------
# backoff_filter: open_ended route, real claim-level filtering logic --
# score_claims monkeypatched so no model loads.
# ---------------------------------------------------------------------------

def test_open_ended_route_drops_low_scoring_claims_only(monkeypatch):
    # Two claims: first well-supported (0.9 >= default threshold 0.35),
    # second unsupported (0.05 < 0.35) -- only the second should be dropped.
    monkeypatch.setattr(ca, "score_claims", lambda claims, context, device="cpu": [0.9, 0.05])
    monkeypatch.setattr(ca, "decompose_claims", lambda answer: ["Claim one.", "Claim two."])
    result = ca.backoff_filter("Claim one. Claim two.", context="some context", route="open_ended")
    assert result["filtered_answer"] == "Claim one."
    assert result["retained_fraction"] == 0.5
    assert len(result["dropped_claims"]) == 1
    assert result["dropped_claims"][0]["claim"] == "Claim two."


def test_open_ended_route_custom_threshold_overrides_default(monkeypatch):
    monkeypatch.setattr(ca, "score_claims", lambda claims, context, device="cpu": [0.5])
    monkeypatch.setattr(ca, "decompose_claims", lambda answer: ["Claim."])
    # 0.5 clears a low custom threshold but would fail the module default (0.35 -- wait,
    # 0.5 > 0.35 too, so use a threshold ABOVE 0.5 to force a drop instead)
    result = ca.backoff_filter("Claim.", context="ctx", route="open_ended", threshold=0.6)
    assert result["retained_fraction"] == 0.0
    assert result["threshold"] == 0.6


# ---------------------------------------------------------------------------
# calibrate_threshold_from_labels: the Learn-Then-Test-style selection logic
# -- verifies it picks the SMALLEST lambda meeting target_risk (see the
# 2026-07-31 docstring fix correcting a self-contradictory description),
# not the largest.
# ---------------------------------------------------------------------------

def test_calibrate_picks_smallest_threshold_meeting_target_risk(monkeypatch):
    # 4 labeled claims with scores 0.1, 0.4, 0.7, 0.9; "b" (score 0.4) is the
    # one INCORRECT claim. Retaining score>=0.1 or >=0.4 both still include
    # "b" -> risk=1/4=0.25 or 1/3=0.333, too high for target_risk=0.1.
    # Retaining score>=0.7 excludes "b" entirely -> risk=0.0, meets target.
    # So the smallest valid lambda should be 0.7, not 0.1/0.4 (too risky,
    # still include the wrong claim) and not 0.9 (more aggressive than
    # needed once 0.7 already excludes "b").
    labeled_rows = [
        {"claim": "a", "context": "ctx", "is_correct": True},
        {"claim": "b", "context": "ctx", "is_correct": False},
        {"claim": "c", "context": "ctx", "is_correct": True},
        {"claim": "d", "context": "ctx", "is_correct": True},
    ]
    scores_by_claim = {"a": 0.1, "b": 0.4, "c": 0.7, "d": 0.9}
    monkeypatch.setattr(ca, "score_claims",
                         lambda claims, context, device="cpu": [scores_by_claim[claims[0]]])
    result = ca.calibrate_threshold_from_labels(labeled_rows, target_risk=0.1)
    assert result["threshold"] == 0.7, f"expected smallest valid lambda 0.7, got {result['threshold']}"
    assert result["achieved_risk"] == 0.0
    assert result["n"] == 4


def test_calibrate_raises_on_empty_calibration_set():
    try:
        ca.calibrate_threshold_from_labels([], target_risk=0.1)
        assert False, "expected ValueError on empty labeled_rows"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# _zigzag_by_confidence (pipeline/novel_pipeline.py) -- hand-verified during
# code review as correct, given a regression test here so a future edit
# can't silently break the "sandwich" ordering without a test failing.
# ---------------------------------------------------------------------------

def test_zigzag_orders_as_sandwich_highest_first_second_highest_last():
    # Descending-score input (already sorted, as build_context produces):
    # 4 (highest), 3, 2, 1 (lowest). Expected sandwich: [4, 2, 1, 3]
    # -- highest first, second-highest last, third-highest second,
    # fourth-highest second-to-last.
    scored = [(4.0, "four"), (3.0, "three"), (2.0, "two"), (1.0, "one")]
    result = _zigzag_by_confidence(scored)
    assert result == ["four", "two", "one", "three"], f"unexpected order: {result}"


def test_zigzag_handles_odd_count():
    scored = [(3.0, "c"), (2.0, "b"), (1.0, "a")]
    result = _zigzag_by_confidence(scored)
    # highest first, second-highest last, third-highest (the only one left) second
    assert result == ["c", "a", "b"], f"unexpected order: {result}"


def test_zigzag_handles_single_item():
    assert _zigzag_by_confidence([(1.0, "only")]) == ["only"]


def test_zigzag_handles_empty_list():
    assert _zigzag_by_confidence([]) == []


if __name__ == "__main__":
    # Plain-invocation path: monkeypatch fixture isn't available outside
    # pytest, so build a minimal stand-in with the same .setattr interface.
    class _FakeMonkeypatch:
        def __init__(self):
            self._restore = []

        def setattr(self, obj, name, value):
            self._restore.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._restore):
                setattr(obj, name, old)

    def _run(fn):
        mp = _FakeMonkeypatch()
        try:
            if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(mp)
            else:
                fn()
        finally:
            mp.undo()

    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _run(_fn)
    print("OK: all conformal_abstention/zigzag unit tests passed")
