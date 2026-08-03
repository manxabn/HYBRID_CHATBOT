"""
Two more genuine attempts to improve on the deployed 4-signal open-ended
abstention classifier (0.644 mean CV accuracy), tried honestly -- not
targeting a specific number, reporting whatever the real result is.

Attempt A -- continuous question_match_any: the deployed classifier uses
a hard-thresholded binary version (>=0.90 ratio -> 1.0, else 0.0),
discarding the actual match-closeness information below/above that cut.
Using the raw max _question_match_ratio value (continuous, [0,1]) instead
gives the classifier strictly more information -- same underlying
signal, no thresholding information loss.

Attempt B -- a shallow gradient-boosted tree instead of logistic
regression, same 4 (or 5, with continuous ratio) features: logistic
regression only captures a LINEAR combination of the features; if the
true decision boundary is mildly nonlinear (e.g., an interaction between
bm25_vector_agreement and query_top1_score), a small tree ensemble could
capture that. Deliberately shallow (max_depth=3, n_estimators=50) given
n=371 -- a large model here would overfit hard, not genuinely generalize.

Same 5-seed x 5-fold robustness protocol as every other abstention-signal
experiment in this project, compared directly against the CURRENTLY
DEPLOYED 4-signal logistic regression (0.644), not a stale baseline.

OFFLINE MEASUREMENT ONLY. Does not touch results/abstention_threshold_
newsignals.json unless a result is a clear, seed-robust improvement.

Usage: python scripts/calibrate_abstention_v2_signals.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.novel_pipeline import _question_match_ratio, QUESTION_MATCH_THRESHOLD
from scripts.calibrate_abstention_newsignals import ROBUSTNESS_SEEDS
from scripts.calibrate_abstention import SAMPLE_SEED, load_rows, build_synthetic_entity_calibration_set
from scripts.calibrate_abstention_kfold import make_folds, N_FOLDS

OUT_PATH = ROOT / "results" / "abstention_threshold_v2_signals.json"
DEPLOYED_FEATURES = ["query_confidence", "query_top1_score", "question_match_any", "bm25_vector_agreement"]
CONTINUOUS_FEATURES = ["query_confidence", "query_top1_score", "question_match_ratio_continuous", "bm25_vector_agreement"]


def compute_signals(retriever: HybridRetriever, query: str, results: list):
    """Returns (question_match_any_binary, question_match_ratio_continuous, bm25_vector_agreement)."""
    max_ratio = 0.0
    for r in results:
        ratio = _question_match_ratio(query, r["text"])
        if ratio > max_ratio:
            max_ratio = ratio
    question_match_any = 1.0 if max_ratio >= QUESTION_MATCH_THRESHOLD else 0.0

    bm25_cand = retriever._bm25_candidates(query)
    vec_cand = retriever._vector_candidates(query)
    bm25_top1 = max(bm25_cand, key=bm25_cand.get) if bm25_cand else None
    vec_top1 = min(vec_cand, key=vec_cand.get) if vec_cand else None
    bm25_vector_agreement = 1.0 if (bm25_top1 is not None and bm25_top1 == vec_top1) else 0.0

    return question_match_any, max_ratio, bm25_vector_agreement


def cv_accuracy(X, labels_arr, clf_factory, seeds):
    all_accs = []
    for seed in seeds:
        folds = make_folds(labels_arr.tolist(), N_FOLDS, seed)
        for k in range(N_FOLDS):
            test_idx = folds[k]
            train_idx = [i for j, f in enumerate(folds) if j != k for i in f]
            clf = clf_factory()
            clf.fit(X[train_idx], labels_arr[train_idx])
            all_accs.append(float((clf.predict(X[test_idx]) == labels_arr[test_idx]).mean()))
    return all_accs


def main():
    out_of_scope, answerable_sample = load_rows()
    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()
    print(f"Out-of-Scope: {len(out_of_scope)}, answerable: {len(answerable_sample)}, "
          f"synthetic: {len(synthetic_queries)}", flush=True)

    retriever = HybridRetriever()
    queries = out_of_scope + answerable_sample + synthetic_queries
    labels = [True] * len(out_of_scope) + [False] * len(answerable_sample) + synthetic_labels

    data = {"labels": [], "query_confidence": [], "query_top1_score": [],
            "question_match_any": [], "question_match_ratio_continuous": [], "bm25_vector_agreement": []}

    t0 = time.perf_counter()
    for i, (q, label) in enumerate(zip(queries, labels)):
        results, meta = retriever.retrieve_adaptive(q)
        if meta["route"] != "open_ended":
            continue
        if results:
            qc = results[0]["query_confidence"]
            qt = results[0]["query_top1_score"]
        else:
            qc, qt = 0.0, 0.0
        qma, qmr, bva = compute_signals(retriever, q, results)
        data["labels"].append(label)
        data["query_confidence"].append(qc)
        data["query_top1_score"].append(qt)
        data["question_match_any"].append(qma)
        data["question_match_ratio_continuous"].append(qmr)
        data["bm25_vector_agreement"].append(bva)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} retrieved ({time.perf_counter()-t0:.1f}s)", flush=True)

    n = len(data["labels"])
    print(f"\nopen_ended n={n}", flush=True)
    labels_arr = np.array(data["labels"], dtype=int)
    X_deployed = np.column_stack([np.array(data[f]) for f in DEPLOYED_FEATURES])
    X_continuous = np.column_stack([np.array(data[f]) for f in CONTINUOUS_FEATURES])

    print("\n--- Baseline: deployed 4-signal logistic regression (binary question_match_any) ---")
    baseline_accs = cv_accuracy(X_deployed, labels_arr, lambda: LogisticRegression(class_weight="balanced", max_iter=1000), ROBUSTNESS_SEEDS)
    print(f"mean={np.mean(baseline_accs):.4f} std={np.std(baseline_accs):.4f}")

    print("\n--- Attempt A: continuous question_match_ratio, logistic regression ---")
    contA_accs = cv_accuracy(X_continuous, labels_arr, lambda: LogisticRegression(class_weight="balanced", max_iter=1000), ROBUSTNESS_SEEDS)
    print(f"mean={np.mean(contA_accs):.4f} std={np.std(contA_accs):.4f}")
    tA, pA = stats.ttest_rel(contA_accs, baseline_accs)
    wA, pwA = stats.wilcoxon(contA_accs, baseline_accs) if not np.allclose(contA_accs, baseline_accs) else (float("nan"), 1.0)
    print(f"Paired t-test vs deployed: p={pA:.4f}  Wilcoxon: p={pwA:.4f}")

    print("\n--- Attempt B: shallow gradient-boosted trees, same 4 deployed features ---")
    contB_accs = cv_accuracy(X_deployed, labels_arr,
                              lambda: GradientBoostingClassifier(max_depth=3, n_estimators=50, random_state=0),
                              ROBUSTNESS_SEEDS)
    print(f"mean={np.mean(contB_accs):.4f} std={np.std(contB_accs):.4f}")
    tB, pB = stats.ttest_rel(contB_accs, baseline_accs)
    wB, pwB = stats.wilcoxon(contB_accs, baseline_accs) if not np.allclose(contB_accs, baseline_accs) else (float("nan"), 1.0)
    print(f"Paired t-test vs deployed: p={pB:.4f}  Wilcoxon: p={pwB:.4f}")

    result = {
        "n": n, "n_folds": N_FOLDS, "seeds": ROBUSTNESS_SEEDS,
        "deployed_4signal_mean_accuracy": round(float(np.mean(baseline_accs)), 4),
        "attemptA_continuous_ratio_mean_accuracy": round(float(np.mean(contA_accs)), 4),
        "attemptA_paired_ttest_p": round(float(pA), 4), "attemptA_paired_wilcoxon_p": round(float(pwA), 4),
        "attemptB_gbt_mean_accuracy": round(float(np.mean(contB_accs)), 4),
        "attemptB_paired_ttest_p": round(float(pB), 4), "attemptB_paired_wilcoxon_p": round(float(pwB), 4),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")

    print("\n=== SUMMARY ===")
    print(f"Deployed:              {np.mean(baseline_accs):.4f}")
    print(f"A (continuous ratio):  {np.mean(contA_accs):.4f}  (p_t={pA:.3f}, p_w={pwA:.3f})")
    print(f"B (gradient boosting): {np.mean(contB_accs):.4f}  (p_t={pB:.3f}, p_w={pwB:.3f})")


if __name__ == "__main__":
    main()
