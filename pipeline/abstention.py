"""
Confidence-gated abstention gate.

hybrid_retriever.py has always computed query_confidence (the top1-top2
fused-score margin) but never used it -- CLAUDE.md.md's own status notes
say explicitly "NO abstention threshold set yet (needs task #2's labeled
data to tune against)". The labeled data it's asking for already exists in
this repo and is already unused: EnglishQA/BanglishQA both carry a
"Category" field with a "Out of Scope / Unanswerable" value (127 EnglishQA
rows, per CLAUDE.md.md's own count) whose reference answers are deliberate
refusals, not facts. Those rows are exactly a should-abstain/should-not-
abstain labeled set -- see scripts/calibrate_abstention.py, which uses them
to pick THRESHOLD below rather than guessing it.

Per-route thresholds, not one global threshold
-------------------------------------------------
Confirmed live (first real run of the novel pipeline, 2026-07-26): a single
global threshold systematically over-abstains on the entity_heavy/RRF route
and under-abstains on open_ended/linear. This is a scale mismatch, not a
sampling artifact -- RRF scores are sums of 1/(60+rank) terms (max ~0.033
for a rank-0 top-1), while linear scores are direct convex combinations in
[0,1], so RRF's top1-top2 margins are structurally an order of magnitude
smaller than linear's even when RRF's retrieval is equally or more
confident. Calibrating one threshold each per route (still against the same
Out-of-Scope-labeled data, just partitioned by which route retrieve_adaptive
picked) is the correct fix -- see scripts/calibrate_abstention.py.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD_PATH = ROOT / "results" / "abstention_threshold.json"

ABSTENTION_MESSAGE = (
    "I don't have reliable information retrieved for this question. "
    "Please check with your academic advisor or BRAC University's official "
    "resources rather than relying on a guess here."
)


class AbstentionGate:
    def __init__(self, config: dict = None):
        """config: {"entity_heavy": {"signal": str, "threshold": float}, "open_ended": {...}}.
        If not given, loads the calibrated per-route signal+threshold
        written by scripts/calibrate_abstention.py. The signal choice
        matters: margin (query_confidence) and raw top-1 score
        (query_top1_score) are both real, independently-evaluated
        candidates -- see calibrate_abstention.py, which picks whichever
        empirically classifies Out-of-Scope-vs-answerable better per route,
        rather than assuming one is universally correct."""
        if config is None:
            config = self._load_calibrated_config()
        self.config = config

    @staticmethod
    def _load_calibrated_config() -> dict:
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "entity_heavy" in data and "open_ended" in data:
                return {route: {"signal": data[route].get("signal", "query_confidence"),
                                "threshold": data[route]["threshold"]}
                        for route in ("entity_heavy", "open_ended")}
            # Backward-compat with the earliest single-threshold calibration
            # file, in case this is run before recalibrating.
            return {"entity_heavy": {"signal": "query_confidence", "threshold": data["threshold"]},
                    "open_ended": {"signal": "query_confidence", "threshold": data["threshold"]}}
        raise FileNotFoundError(
            f"{THRESHOLD_PATH} not found. Run "
            "`python scripts/calibrate_abstention.py` first, or pass an "
            "explicit config=... to AbstentionGate()."
        )

    def should_abstain(self, signals: dict, route: str) -> bool:
        """signals: {"query_confidence": float, "query_top1_score": float}
        (both fields hybrid_retriever.py's retrieve()/retrieve_adaptive()
        results now carry)."""
        route_config = self.config.get(route)
        if route_config is None:
            raise ValueError(f"No calibrated config for route={route!r}")
        value = signals[route_config["signal"]]
        return value < route_config["threshold"]
