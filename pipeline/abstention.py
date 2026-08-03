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


CLASSIFIER_THRESHOLD_PATH = ROOT / "results" / "abstention_threshold_newsignals.json"
GBT_MODEL_PATH = ROOT / "results" / "abstention_gbt_open_ended.joblib"
GBT_META_PATH = ROOT / "results" / "abstention_gbt_open_ended_metadata.json"


class AbstentionGate:
    def __init__(self, config: dict = None):
        """config: {"entity_heavy": {"signal": str, "threshold": float}, "open_ended": {...}}.
        If not given, loads the calibrated per-route signal+threshold
        written by scripts/calibrate_abstention.py. The signal choice
        matters: margin (query_confidence) and raw top-1 score
        (query_top1_score) are both real, independently-evaluated
        candidates -- see calibrate_abstention.py, which picks whichever
        empirically classifies Out-of-Scope-vs-answerable better per route,
        rather than assuming one is universally correct.

        open_ended route (2026-08-02): a small 4-feature logistic
        regression (query_confidence, query_top1_score, question_match_any,
        bm25_vector_agreement) rather than a single-signal threshold --
        see scripts/calibrate_abstention_newsignals.py for why: the two new
        signals are structurally different from the two score-magnitude
        signals (a lexical near-match check and a cross-retrieval-stream
        agreement check), and combining all four via 5-fold cross-validated
        logistic regression measured a consistent +5.2pp mean accuracy
        improvement over the single-threshold gate (5/5 independent fold
        -assignment seeds won, not a lucky split -- see that script's
        robustness sweep). entity_heavy is untouched: its single-threshold
        gate already generalizes at 96.7% CV accuracy (results/abstention_
        threshold_kfold.json), nothing to fix there.

        open_ended route (2026-08-03, supersedes the logistic regression
        above when its artifact is present): a shallow gradient-boosted
        tree ensemble (max_depth=3, n_estimators=50) over the SAME 4
        features, found to substantially outperform logistic regression
        -- 0.644 -> 0.737 mean CV accuracy, verified robust before
        deploying: beats the logistic regression in every one of 5
        independent fold-assignment seeds (paired t-test AND Wilcoxon
        signed-rank both p<0.0001 -- this project's standing bar of both
        tests agreeing), stable across 5 different classifier random
        seeds (0.7374-0.7390), and a moderate, non-suspicious train/CV
        gap (0.873 in-sample vs. 0.737 cross-validated -- not the
        near-100%/near-chance pattern that would indicate memorization).
        See scripts/calibrate_abstention_v2_signals.py (the comparison)
        and scripts/calibrate_abstention_gbt_final.py (the deployed
        artifact). Loaded via joblib -- a self-generated artifact from
        this project's own data, not a third-party pickle downloaded
        from the internet, a materially different trust profile from the
        externally-hosted checkpoints declined elsewhere this session."""
        if config is None:
            config = self._load_calibrated_config()
        self.config = config

    @staticmethod
    def _load_calibrated_config() -> dict:
        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                f"{THRESHOLD_PATH} not found. Run "
                "`python scripts/calibrate_abstention.py` first, or pass an "
                "explicit config=... to AbstentionGate()."
            )
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "entity_heavy" not in data or "open_ended" not in data:
            # Backward-compat with the earliest single-threshold calibration
            # file, in case this is run before recalibrating.
            return {"entity_heavy": {"signal": "query_confidence", "threshold": data["threshold"]},
                    "open_ended": {"signal": "query_confidence", "threshold": data["threshold"]}}

        config = {"entity_heavy": {"signal": data["entity_heavy"].get("signal", "query_confidence"),
                                    "threshold": data["entity_heavy"]["threshold"]}}

        if GBT_MODEL_PATH.exists() and GBT_META_PATH.exists():
            import joblib
            with open(GBT_META_PATH, "r", encoding="utf-8") as f:
                gbt_meta = json.load(f)
            config["open_ended"] = {
                "model": "gradient_boosting",
                "features": gbt_meta["features"],
                "sklearn_model": joblib.load(GBT_MODEL_PATH),
            }
        elif CLASSIFIER_THRESHOLD_PATH.exists():
            with open(CLASSIFIER_THRESHOLD_PATH, "r", encoding="utf-8") as f:
                clf_data = json.load(f)
            config["open_ended"] = {
                "model": "logistic_regression",
                "features": clf_data["features"],
                "coefficients": clf_data["full_data_coefficients"],
                "intercept": clf_data["full_data_intercept"],
            }
        else:
            config["open_ended"] = {"signal": data["open_ended"].get("signal", "query_confidence"),
                                     "threshold": data["open_ended"]["threshold"]}
        return config

    def should_abstain(self, signals: dict, route: str) -> bool:
        """signals: dict carrying at least "query_confidence" and
        "query_top1_score" (both fields hybrid_retriever.py's retrieve()/
        retrieve_adaptive() results now carry); the open_ended classifier
        additionally needs "question_match_any" and "bm25_vector_agreement"
        (see novel_pipeline.py's build_context, which computes both before
        calling this)."""
        route_config = self.config.get(route)
        if route_config is None:
            raise ValueError(f"No calibrated config for route={route!r}")
        if route_config.get("model") == "gradient_boosting":
            x = [[signals[f] for f in route_config["features"]]]
            return bool(route_config["sklearn_model"].predict(x)[0])
        if route_config.get("model") == "logistic_regression":
            z = route_config["intercept"] + sum(
                route_config["coefficients"][f] * signals[f] for f in route_config["features"]
            )
            return z >= 0.0  # matches sklearn LogisticRegression.predict(): class 1 (should-abstain) iff decision_function >= 0
        value = signals[route_config["signal"]]
        return value < route_config["threshold"]
