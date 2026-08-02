"""
A genuinely untried angle on the open-ended abstention gate's weak
generalization (CV accuracy 0.6118, results/abstention_threshold_kfold.json)
-- distinct from the already-tried-and-ruled-out 3-signal logistic
regression (2026-08-01, CLAUDE.md.md), which combined three signals from
the SAME family (query_top1_score, max(s_vec), query_confidence -- all
retrieval-score magnitudes) on the narrower 80-query paraphrase-robustness
sample and found they all trend in the wrong direction. That result rules
out "recombine the score magnitudes" as a fix; it says nothing about
signals of a structurally different kind.

This script adds two NEW signals, neither of which is a retrieval-score
magnitude, evaluated on the actual deployed open-ended calibration set
(n=371, same queries/labels as calibrate_abstention_kfold.py):

  - question_match_any: whether ANY retrieved candidate's own source
    question (the "Q:" line stored with question-style corpus rows, see
    novel_pipeline.py's QUESTION_LINE_RE / _question_match_ratio) is a
    near-verbatim string match (ratio >= QUESTION_MATCH_THRESHOLD=0.90) to
    the query. Currently only used one-directionally in novel_pipeline.py
    as a sufficient-context override; never tried as an independent
    abstention-confidence feature. A structural/lexical-match signal, not
    a score magnitude.
  - bm25_vector_agreement: whether BM25's own top-1 candidate and the
    vector stream's own top-1 candidate (before fusion) are the SAME
    document. The intuition: when two independent retrieval mechanisms
    agree on the single best candidate, that agreement is itself
    information the fused top1_score/margin can't see (a fused score can
    be high even when the two streams disagree about WHICH doc is best,
    if one stream's high confidence dominates the blend). Also
    structurally different from a score magnitude.

Both are computed directly from the SAME underlying retrieval call
already made for query_confidence/query_top1_score, no extra retrieval
work. Combined with the existing two signals via a 5-fold cross-validated
logistic regression (sklearn, class_weight='balanced', same fold
assignment as calibrate_abstention_kfold.py's make_folds so the CV
estimate is directly comparable to the deployed 0.6118 baseline).

This is an OFFLINE MEASUREMENT ONLY. It does not touch results/
abstention_threshold.json (the deployed threshold) or any pipeline code.
If the result is not a clear, honest improvement, it is reported as such
and nothing is deployed -- same discipline as every other calibration
experiment in this project's history.

Usage: python scripts/calibrate_abstention_newsignals.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.novel_pipeline import _question_match_ratio, QUESTION_MATCH_THRESHOLD
from scripts.calibrate_abstention import (
    SAMPLE_SEED, SIGNALS, load_rows, build_synthetic_entity_calibration_set,
    precision_recall_f1, clopper_pearson_ci,
)
from scripts.calibrate_abstention_kfold import make_folds, sweep_best, N_FOLDS, SPLIT_SEED

OUT_PATH = ROOT / "results" / "abstention_threshold_newsignals.json"
NEW_FEATURES = ["query_confidence", "query_top1_score", "question_match_any", "bm25_vector_agreement"]

# Robustness check (added after the first single-seed run showed a +0.046
# CV accuracy gain, before trusting it enough to even consider deployment):
# a single fold assignment can make any classifier look better or worse by
# chance, especially at n=371 with only 5 folds -- the deployed baseline
# itself was only ever validated at SPLIT_SEED=42. Re-running BOTH the
# existing single-threshold baseline AND the new 4-signal classifier under
# several independent fold assignments, and comparing their means under
# the SAME seeds, is the only way to tell "this is a real, stable gain"
# apart from "this seed happened to flatter the classifier."
ROBUSTNESS_SEEDS = [42, 1, 7, 123, 2026]


def compute_new_signals(retriever: HybridRetriever, query: str, results: list) -> tuple[float, float]:
    """Returns (question_match_any, bm25_vector_agreement) as 0.0/1.0 floats."""
    question_match_any = 0.0
    for r in results:
        if _question_match_ratio(query, r["text"]) >= QUESTION_MATCH_THRESHOLD:
            question_match_any = 1.0
            break

    bm25_cand = retriever._bm25_candidates(query)
    vec_cand = retriever._vector_candidates(query)
    bm25_top1 = max(bm25_cand, key=bm25_cand.get) if bm25_cand else None
    vec_top1 = min(vec_cand, key=vec_cand.get) if vec_cand else None  # distance: lower is better
    bm25_vector_agreement = 1.0 if (bm25_top1 is not None and bm25_top1 == vec_top1) else 0.0

    return question_match_any, bm25_vector_agreement


def main():
    out_of_scope, answerable_sample = load_rows()
    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()
    print(f"Out-of-Scope: {len(out_of_scope)}, answerable: {len(answerable_sample)}, "
          f"synthetic: {len(synthetic_queries)}", flush=True)

    retriever = HybridRetriever()
    queries = out_of_scope + answerable_sample + synthetic_queries
    labels = [True] * len(out_of_scope) + [False] * len(answerable_sample) + synthetic_labels

    data = {"labels": [], "query_confidence": [], "query_top1_score": [],
            "question_match_any": [], "bm25_vector_agreement": []}
    route_of = []

    t0 = time.perf_counter()
    for i, (q, label) in enumerate(zip(queries, labels)):
        results, meta = retriever.retrieve_adaptive(q)
        route_of.append(meta["route"])
        if meta["route"] != "open_ended":
            continue
        if results:
            qc = results[0]["query_confidence"]
            qt = results[0]["query_top1_score"]
        else:
            qc, qt = 0.0, 0.0
        qma, bva = compute_new_signals(retriever, q, results)
        data["labels"].append(label)
        data["query_confidence"].append(qc)
        data["query_top1_score"].append(qt)
        data["question_match_any"].append(qma)
        data["bm25_vector_agreement"].append(bva)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} retrieved ({time.perf_counter()-t0:.1f}s)", flush=True)

    n = len(data["labels"])
    print(f"\nopen_ended n={n} (must match abstention_threshold_kfold.json's n=371 "
          f"for the CV comparison below to be apples-to-apples)", flush=True)

    labels_arr = np.array(data["labels"], dtype=int)
    X = np.column_stack([np.array(data[f]) for f in NEW_FEATURES])

    print(f"question_match_any: {int(data['question_match_any'].count(1.0))}/{n} true "
          f"(base rate; a signal that's ~always 0 or ~always 1 can't discriminate anything)")
    print(f"bm25_vector_agreement: {int(sum(data['bm25_vector_agreement']))}/{n} true", flush=True)

    per_seed = []
    all_clf_accs = []
    all_baseline_accs = []
    for seed in ROBUSTNESS_SEEDS:
        folds = make_folds(data["labels"], N_FOLDS, seed)
        clf_fold_accs = []
        baseline_fold_accs = []
        for k in range(N_FOLDS):
            test_idx = folds[k]
            train_idx = [i for j, f in enumerate(folds) if j != k for i in f]

            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X[train_idx], labels_arr[train_idx])
            pred = clf.predict(X[test_idx])
            clf_fold_accs.append(float((pred == labels_arr[test_idx]).mean()))

            # Baseline re-run under the SAME fold assignment: sweep both
            # original signals on train, evaluate the winner on test --
            # identical procedure to calibrate_abstention_kfold.py, just
            # re-executed here so the comparison is seed-matched rather
            # than pulled from a different (single-seed) stored run.
            route_best_train = None
            for signal_name in SIGNALS:
                best = sweep_best(data["labels"], data[signal_name], train_idx)
                if route_best_train is None or best["accuracy"] > route_best_train["accuracy"]:
                    route_best_train = {**best, "signal": signal_name}
            test_labels = [data["labels"][i] for i in test_idx]
            test_conf = [data[route_best_train["signal"]][i] for i in test_idx]
            test_metrics = precision_recall_f1(test_labels, test_conf, route_best_train["threshold"])
            baseline_fold_accs.append(test_metrics["accuracy"])

        seed_clf_mean = float(np.mean(clf_fold_accs))
        seed_baseline_mean = float(np.mean(baseline_fold_accs))
        per_seed.append({"seed": seed, "clf_cv_accuracy": round(seed_clf_mean, 4),
                          "baseline_cv_accuracy": round(seed_baseline_mean, 4)})
        all_clf_accs.extend(clf_fold_accs)
        all_baseline_accs.extend(baseline_fold_accs)
        print(f"  seed={seed}: 4-signal clf={seed_clf_mean:.4f}  2-signal baseline={seed_baseline_mean:.4f}", flush=True)

    cv_mean_acc = float(np.mean(all_clf_accs))
    cv_std_acc = float(np.std(all_clf_accs))
    baseline_mean_acc = float(np.mean(all_baseline_accs))
    baseline_std_acc = float(np.std(all_baseline_accs))
    n_wins = sum(1 for row in per_seed if row["clf_cv_accuracy"] > row["baseline_cv_accuracy"])

    # Full-data fit, for reporting coefficients (which features the model
    # actually leaned on) -- not used for the CV estimate itself.
    full_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    full_clf.fit(X, labels_arr)
    coefficients = dict(zip(NEW_FEATURES, [float(c) for c in full_clf.coef_[0]]))
    intercept = float(full_clf.intercept_[0])
    full_data_acc = float(full_clf.score(X, labels_arr))

    result = {
        "n": n,
        "n_folds": N_FOLDS,
        "robustness_seeds": ROBUSTNESS_SEEDS,
        "sample_seed": SAMPLE_SEED,
        "features": NEW_FEATURES,
        "per_seed": per_seed,
        "clf_cv_mean_accuracy_across_all_seeds_folds": round(cv_mean_acc, 4),
        "clf_cv_std_accuracy_across_all_seeds_folds": round(cv_std_acc, 4),
        "baseline_cv_mean_accuracy_across_all_seeds_folds": round(baseline_mean_acc, 4),
        "baseline_cv_std_accuracy_across_all_seeds_folds": round(baseline_std_acc, 4),
        "seeds_where_clf_beat_baseline": f"{n_wins}/{len(ROBUSTNESS_SEEDS)}",
        "full_data_coefficients": coefficients,
        "full_data_intercept": intercept,
        "full_data_in_sample_accuracy": round(full_data_acc, 4),
        "question_match_any_base_rate": data["question_match_any"].count(1.0) / n if n else 0.0,
        "bm25_vector_agreement_base_rate": sum(data["bm25_vector_agreement"]) / n if n else 0.0,
        "improvement_over_baseline": round(cv_mean_acc - baseline_mean_acc, 4),
    }

    delta = cv_mean_acc - baseline_mean_acc
    print(f"\nAcross {len(ROBUSTNESS_SEEDS)} independent fold-assignment seeds x {N_FOLDS} folds "
          f"({len(all_clf_accs)} total test folds each):")
    print(f"  4-signal logistic regression: mean={cv_mean_acc:.4f} std={cv_std_acc:.4f}")
    print(f"  2-signal single-threshold (existing method, re-run seed-matched): "
          f"mean={baseline_mean_acc:.4f} std={baseline_std_acc:.4f}")
    print(f"  4-signal classifier beat the baseline in {n_wins}/{len(ROBUSTNESS_SEEDS)} seeds, "
          f"mean delta {delta:+.4f}")
    if n_wins == len(ROBUSTNESS_SEEDS) and delta > 0.02:
        print(f"  Consistent, non-trivial win across every seed tested -- this is a genuine "
              f"signal, not a lucky fold split.")
    elif delta > 0.02 and n_wins >= len(ROBUSTNESS_SEEDS) - 1:
        print(f"  Wins in most seeds with a non-trivial mean delta -- a real but not "
              f"unanimous improvement.")
    else:
        print(f"  Not consistent/large enough across seeds to call a reliable improvement.")
    print(f"Full-data logistic regression coefficients: {coefficients}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH} (measurement only -- deployed abstention_threshold.json untouched)")


if __name__ == "__main__":
    main()
