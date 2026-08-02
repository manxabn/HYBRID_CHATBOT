"""
Unit tests for pipeline/abstention.py -- previously untested despite being
a live safety-relevant gate. Covers both the entity_heavy route's
unchanged single-threshold path and the open_ended route's new (2026-08-02)
4-feature logistic-regression classifier path (see scripts/calibrate_
abstention_newsignals.py for how the classifier was fit and validated).

Plain asserts, runnable directly: `python tests/test_abstention.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.abstention import AbstentionGate


# ---------------------------------------------------------------------------
# Legacy single-threshold path (entity_heavy, and open_ended when no
# classifier config is supplied) -- must behave exactly as before.
# ---------------------------------------------------------------------------

def _threshold_only_config():
    return {"entity_heavy": {"signal": "query_confidence", "threshold": 1.0},
            "open_ended": {"signal": "query_top1_score", "threshold": 0.5}}


def test_threshold_path_abstains_below_threshold():
    gate = AbstentionGate(config=_threshold_only_config())
    assert gate.should_abstain({"query_confidence": 0.5}, "entity_heavy") is True


def test_threshold_path_does_not_abstain_at_or_above_threshold():
    gate = AbstentionGate(config=_threshold_only_config())
    assert gate.should_abstain({"query_confidence": 1.0}, "entity_heavy") is False
    assert gate.should_abstain({"query_confidence": 2.0}, "entity_heavy") is False


def test_unknown_route_raises():
    gate = AbstentionGate(config=_threshold_only_config())
    try:
        gate.should_abstain({"query_confidence": 1.0}, "not_a_route")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Classifier path (open_ended, 2026-08-02) -- a small logistic regression
# over 4 features. Uses simple, hand-checkable coefficients rather than the
# real fitted ones, so the test verifies the DECISION RULE (sigmoid/
# decision_function >= 0 => abstain), not the specific fitted numbers
# (those live in results/abstention_threshold_newsignals.json and are
# re-derived by scripts/calibrate_abstention_newsignals.py, not hardcoded
# here -- hardcoding them would make this test break on every honest
# recalibration).
# ---------------------------------------------------------------------------

def _classifier_config():
    return {
        "open_ended": {
            "model": "logistic_regression",
            "features": ["query_confidence", "query_top1_score",
                         "question_match_any", "bm25_vector_agreement"],
            # intercept=0, single feature with coefficient -1: z = -1 * x.
            # x=1 -> z=-1 (< 0, don't abstain); x=-1 -> z=1 (>= 0, abstain).
            "coefficients": {"query_confidence": -1.0, "query_top1_score": 0.0,
                              "question_match_any": 0.0, "bm25_vector_agreement": 0.0},
            "intercept": 0.0,
        }
    }


def test_classifier_path_decision_boundary_at_zero():
    gate = AbstentionGate(config=_classifier_config())
    signals_high = {"query_confidence": 1.0, "query_top1_score": 0.0,
                     "question_match_any": 0.0, "bm25_vector_agreement": 0.0}
    signals_low = {"query_confidence": -1.0, "query_top1_score": 0.0,
                    "question_match_any": 0.0, "bm25_vector_agreement": 0.0}
    assert gate.should_abstain(signals_high, "open_ended") is False
    assert gate.should_abstain(signals_low, "open_ended") is True


def test_classifier_path_uses_all_four_features():
    # intercept=0, all four coefficients=-1: z = -(sum of features). Any
    # single feature going from 0 to a large positive value should be able
    # to flip the decision from abstain to not-abstain, proving all four
    # are actually read (a stale/typo'd feature key would silently exclude
    # one from the sum and this test would still pass with only 3 -- so
    # flip each feature independently and check each one matters).
    config = {"open_ended": {
        "model": "logistic_regression",
        "features": ["query_confidence", "query_top1_score",
                     "question_match_any", "bm25_vector_agreement"],
        "coefficients": {"query_confidence": -1.0, "query_top1_score": -1.0,
                          "question_match_any": -1.0, "bm25_vector_agreement": -1.0},
        "intercept": 0.0,
    }}
    gate = AbstentionGate(config=config)
    zeros = {"query_confidence": 0.0, "query_top1_score": 0.0,
             "question_match_any": 0.0, "bm25_vector_agreement": 0.0}
    assert gate.should_abstain(dict(zeros), "open_ended") is True  # z=0 >= 0 -> abstain
    for feature in config["open_ended"]["features"]:
        signals = dict(zeros)
        signals[feature] = 5.0  # pushes z well below 0
        assert gate.should_abstain(signals, "open_ended") is False, \
            f"flipping {feature} alone should have flipped the decision"


def test_real_deployed_config_loads_and_returns_bool():
    # Integration check against the actual deployed artifacts (results/
    # abstention_threshold.json + abstention_threshold_newsignals.json) --
    # skipped gracefully if the repo hasn't been calibrated in this
    # checkout, so this file stays runnable on a fresh clone.
    threshold_path = ROOT / "results" / "abstention_threshold.json"
    if not threshold_path.exists():
        print("SKIP: results/abstention_threshold.json not present, skipping integration check")
        return
    gate = AbstentionGate()
    assert isinstance(gate.config["entity_heavy"], dict)
    assert isinstance(gate.config["open_ended"], dict)
    signals = {"query_confidence": 0.5, "query_top1_score": 0.5,
               "question_match_any": 1.0, "bm25_vector_agreement": 1.0}
    for route in ("entity_heavy", "open_ended"):
        result = gate.should_abstain(signals, route)
        assert isinstance(result, bool)


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("OK: all abstention.py unit tests passed")
