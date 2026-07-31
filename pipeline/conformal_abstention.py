"""
Claim-level back-off for generated answers, structurally following Mohri &
Hashimoto, "Language Models with Conformal Factuality Guarantees" (ICML 2024,
proceedings.mlr.press/v235/mohri24a.html -- independently verified via
WebFetch before this citation was used, per this project's standing
zero-fabrication rule). Their method: treat an answer's correctness as an
uncertainty-quantification problem over the entailment set of its individual
claims, and progressively remove (back off) the least-supported claims until
the remaining ones clear a calibrated confidence threshold, giving a
high-probability correctness guarantee for whatever remains.

This module implements the MECHANISM (claim decomposition + entailment
scoring + threshold back-off), reusing this project's own existing,
independent NLI entailment infrastructure (scripts/measure_nli_faithfulness.py's
SummaC-ZS sentence-level max-aggregation, Laban et al. 2022 TACL) as the
per-claim confidence signal, rather than re-implementing entailment scoring
from scratch.

WHAT THIS DOES NOT YET DO: provide the calibrated statistical GUARANTEE the
original paper's method depends on. That requires a held-out calibration set
of (claim, context, human-judged correct/incorrect) labels -- exactly the
gap this project's own standing limitations list already names ("no human
hallucination annotation yet"). Until that calibration set exists and is
labeled (results/human_annotation_sample.csv, prepared earlier this project
with 50 disagreement-prioritized rows and blank human_label columns, is the
natural candidate), OPEN_ENDED_THRESHOLD below is a disclosed, provisional
value, NOT a calibrated one -- calibrate_threshold_from_labels() is provided
specifically so this can be replaced with a real calibration the moment
human labels exist, without changing anything else in this module. Treating
the provisional threshold as if it already carried Mohri & Hashimoto's
80-90% guarantee would itself be exactly the kind of unverified claim this
project's standing rule exists to prevent -- so it is not claimed here.
"""

import sys
from pathlib import Path

import nltk
import numpy as np
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
LABELS = ["contradiction", "entailment", "neutral"]

PROVISIONAL_NOTE = (
    "threshold is provisional (heuristic, derived from this project's own "
    "prior NLI faithfulness distribution), not yet calibrated against "
    "human-labeled correctness data"
)

# Route-specific behavior (2026-07-28, found by direct testing before this
# module was ever wired into the live pipeline -- not assumed): applying
# DEFAULT_THRESHOLD uniformly caused the pipeline to abstain on completely
# correct, exact-match-grounded answers like "Dr. Mohammad Kaykobad's office
# room is 4G11" (NLI score 0.099) purely because the NLI cross-encoder,
# trained on natural sentence-pair entailment (SNLI/MultiNLI), does not
# recognize that structured field:value context ("Name: Dr. Mohammad
# Kaykobad ... Room: 4G11") entails a natural-language restatement of it --
# a format mismatch, not a real faithfulness problem. Checking this
# project's own already-computed nli_faithfulness_per_query.csv (n=170,
# from an earlier faithfulness measurement pass) confirms this is
# systematic, not a fluke: entity_heavy-route rows score mean=0.251,
# median=0.106; open_ended-route rows score mean=0.744, median=0.958 -- a
# stark bimodal split that lines up exactly with route, not with any
# plausible real difference in answer correctness (entity-heavy answers are
# in fact this system's MOST reliable ones, per the unambiguous-match
# ceiling). Applying one threshold across both routes would systematically
# false-abstain on the entity-heavy route specifically -- so this module
# only applies claim-level NLI back-off to open_ended-route answers, and
# for entity_heavy-route answers defers entirely to the exact_match_any
# signal novel_pipeline.py's build_context already computes (already a
# stronger, validated grounding signal for this route than NLI entailment
# is or would be).
ENTITY_HEAVY_THRESHOLD = None  # sentinel: skip claim-level back-off for this route, see above

# OPEN_ENDED_THRESHOLD is even more provisional than the module docstring
# already disclosed, per a second real finding from the same distribution
# check: several open_ended rows the LLM-judge scored fully faithful
# (faithfulness=1.0) still get NLI scores of 0.013-0.15 -- e.g. "To submit
# a report after a club event, you need to send an email with..." scored
# LLM-judge=1.0 but NLI=0.069. This is the same weak LLM-judge/NLI
# correlation already found and reported this project (Pearson r=0.089,
# results/nli_vs_llmjudge_agreement.csv), now showing up as a concrete
# false-positive-backoff risk: a fixed threshold here (even one restricted
# to open_ended only, after the entity_heavy fix above) would still
# incorrectly flag roughly 15-20% of genuinely faithful open-ended answers
# as unsupported, based on this project's own measured open_ended score
# percentiles (10th=0.067, 15th=0.102, 20th=0.454 -- the threshold would
# need to sit below the 15th percentile to avoid this, which defeats the
# purpose of the check). This is not a bug to fix by picking a different
# number; it is the underlying NLI entailment signal itself not being
# reliable enough, on its own, to gate real answers without calibration
# against real human-labeled correctness data -- exactly the standing "no
# human hallucination annotation yet" gap this module's docstring already
# names. 0.35 is kept as a documented, disclosed placeholder for what
# calibrate_threshold_from_labels() should replace, NOT a value we
# recommend trusting as-is; use_conformal_backoff remains off by default
# for exactly this reason, verified rather than assumed.
OPEN_ENDED_THRESHOLD = 0.35

_model_cache = {}


def _get_model(device: str = "cpu") -> CrossEncoder:
    # Cached at module level so repeated calls (e.g. scoring many answers in
    # one evaluation run) don't reload a ~440MB model each time -- the
    # "efficient" concern raised alongside "solve every weakness."
    if device not in _model_cache:
        for pkg in ["punkt", "punkt_tab"]:
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        _model_cache[device] = CrossEncoder(NLI_MODEL, device=device)
    return _model_cache[device]


def decompose_claims(answer: str) -> list[str]:
    """Sentence-level claim decomposition -- the same granularity
    scripts/measure_nli_faithfulness.py already uses, so scores from this
    module and that script's descriptive measurements stay directly
    comparable (same method, applied at generation time vs. after the
    fact)."""
    sents = nltk.sent_tokenize(str(answer))
    return sents if sents else [str(answer)]


def score_claims(claims: list[str], context: str, device: str = "cpu") -> list[float]:
    """Per-claim entailment confidence: max entailment probability over all
    context sentences (SummaC-ZS), one score per claim in `claims`, same
    order. This is the nonconformity/confidence signal the back-off
    algorithm removes claims by, ranked lowest-first."""
    model = _get_model(device)
    context_sents = nltk.sent_tokenize(str(context)) or [""]
    pairs = [(c_sent, claim) for claim in claims for c_sent in context_sents]
    if not pairs:
        return []
    with torch.no_grad():
        logits = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    p_entail = probs[:, LABELS.index("entailment")]
    scores = []
    n_context = len(context_sents)
    for i in range(len(claims)):
        seg = p_entail[i * n_context: (i + 1) * n_context]
        scores.append(float(seg.max()) if len(seg) else 0.0)
    return scores


def backoff_filter(answer: str, context: str, route: str = "open_ended",
                    exact_match_any: bool = False, threshold: float = None,
                    device: str = "cpu") -> dict:
    """Applies the back-off: decompose the answer into claims, score each
    against the retrieved context, and DROP any claim whose entailment
    score falls below `threshold`. Returns a dict with the filtered answer
    (only well-supported claims), which claims were dropped and why, and
    the retained fraction -- so a caller (e.g. novel_pipeline.py) can decide
    to show the filtered answer, or treat a very low retained fraction as
    "not enough of this answer is grounded, abstain instead of showing a
    partial answer," mirroring how the existing confidence gate already
    treats "too little support" as a reason to decline.

    route/exact_match_any: see ENTITY_HEAVY_THRESHOLD's module-level
    comment for why entity_heavy routes skip NLI claim-level scoring
    entirely (the NLI model's own blind spot for structured field:value
    context, confirmed via this project's own prior measurements) and
    instead trust exact_match_any -- already computed by novel_pipeline.py
    for every query as its own, more reliable grounding signal for this
    route -- rather than a threshold this module cannot score meaningfully
    for that content type.
    """
    if route == "entity_heavy":
        # Trust the structural exact-match signal, not NLI entailment,
        # for this route -- see module-level comment. A confirmed exact
        # match is treated as fully retained; its absence is treated as
        # fully unretained, matching how sufficient_context_override
        # already uses this exact signal elsewhere in the pipeline.
        retained = 1.0 if exact_match_any else 0.0
        return {
            "filtered_answer": answer if exact_match_any else "",
            "original_answer": answer,
            "claim_scores": [], "dropped_claims": [],
            "retained_fraction": retained,
            "threshold": None,
            "note": "entity_heavy route: deferred to exact_match_any, NLI claim scoring skipped (see module docstring)",
        }

    threshold = OPEN_ENDED_THRESHOLD if threshold is None else threshold
    if not context or not str(context).strip():
        return {
            "filtered_answer": answer, "original_answer": answer,
            "claim_scores": [], "dropped_claims": [], "retained_fraction": 0.0,
            "threshold": threshold, "note": "no context to check claims against; not filtered",
        }
    claims = decompose_claims(answer)
    scores = score_claims(claims, context, device=device)
    kept, dropped = [], []
    for claim, score in zip(claims, scores):
        if score >= threshold:
            kept.append(claim)
        else:
            dropped.append({"claim": claim, "score": round(score, 4)})
    retained_fraction = len(kept) / len(claims) if claims else 0.0
    return {
        "filtered_answer": " ".join(kept) if kept else "",
        "original_answer": answer,
        "claim_scores": [round(s, 4) for s in scores],
        "dropped_claims": dropped,
        "retained_fraction": retained_fraction,
        "threshold": threshold,
        "note": PROVISIONAL_NOTE,
    }


def calibrate_threshold_from_labels(labeled_rows: list[dict], target_risk: float = 0.1,
                                     device: str = "cpu") -> dict:
    """Learn-Then-Test-style calibration (Angelopoulos et al.'s conformal
    risk control recipe, the standard mechanism Mohri & Hashimoto's method
    builds on): given a labeled calibration set, find the SMALLEST threshold
    lambda such that the empirical fraction of calibration examples where a
    RETAINED claim (score >= lambda) was actually human-labeled incorrect
    stays at or below target_risk. Larger lambda = more aggressive removal
    = safer (fewer wrong claims slip through) but retains less of each
    answer; this picks the least-aggressive (smallest) lambda that still
    meets the target, following the standard conservative-calibration
    convention -- assuming risk is (roughly) monotonically non-increasing
    in lambda, which is why candidate_thresholds is scanned ascending and
    the loop stops at the first lambda that meets target_risk, rather than
    continuing to scan for a larger one.

    labeled_rows: list of {"claim": str, "context": str, "is_correct": bool}
    -- one row per individual CLAIM (not per whole answer), human-judged
    against the context it was actually generated from. This is exactly
    the granularity results/human_annotation_sample.csv would need to be
    extended to (currently whole-answer human_label columns, blank,
    prepared but not filled in) -- see this function's caller/CLI for how
    to build labeled_rows from that file once annotated.

    Returns {"threshold": float, "achieved_risk": float, "n": int} -- NOT a
    guarantee by itself; report achieved_risk and n alongside the threshold
    always, since a calibration set of, say, n=30 claims gives a much
    weaker guarantee than n=300, and this function does not hide that."""
    if not labeled_rows:
        raise ValueError("calibrate_threshold_from_labels requires a non-empty labeled calibration set")

    scored = []
    for row in labeled_rows:
        s = score_claims([row["claim"]], row["context"], device=device)
        scored.append((s[0] if s else 0.0, bool(row["is_correct"])))

    candidate_thresholds = sorted({round(s, 4) for s, _ in scored})
    best_threshold = 1.0  # most conservative fallback: drop everything
    best_risk = 0.0
    for lam in candidate_thresholds:
        retained = [(s, correct) for s, correct in scored if s >= lam]
        if not retained:
            continue
        risk = sum(1 for _, correct in retained if not correct) / len(retained)
        if risk <= target_risk:
            best_threshold = lam
            best_risk = risk
            break  # candidate_thresholds is ascending; first hit is the least-aggressive valid lambda

    return {"threshold": best_threshold, "achieved_risk": best_risk, "n": len(labeled_rows)}
