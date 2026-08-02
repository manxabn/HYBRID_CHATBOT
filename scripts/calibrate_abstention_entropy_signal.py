"""
A further attempt on the open-ended abstention gate, after the deployed
4-signal classifier (calibrate_abstention_newsignals.py, 0.612 -> 0.644
mean CV accuracy) -- tests whether a FIFTH, structurally different signal
pushes it further, with the same seed-robustness discipline before
touching anything deployed.

score_entropy: normalized Shannon entropy of the softmax over the full
scored candidate pool (computed in hybrid_retriever.py's retrieve(),
2026-08-03, currently an unused/experimental field -- see that file's
docstring). Distinct from every signal tried so far:
  - query_confidence / query_top1_score: single-number score magnitudes
    (top1 value, or top1-top2 difference).
  - question_match_any / bm25_vector_agreement: binary structural checks.
  - score_entropy: a continuous summary of the WHOLE score distribution's
    shape (0=one clear winner, 1=uniform/no discrimination) -- can
    distinguish cases margin/top1_score cannot, e.g. "one clear winner
    among 15 candidates" (low entropy, high top1) vs. "everything roughly
    tied at a high value" (high entropy, ALSO high top1 -- margin catches
    this via the top-2 gap, but entropy is a genuinely different way of
    summarizing the same underlying distribution that may generalize
    differently out-of-sample).

Standard selective-prediction construct, not invented here (Hendrycks &
Gimpel 2017 softmax-entropy confidence; Kamath, Jia & Liang 2020,
"Selective Question Answering under Domain Shift", uses this exact family
of signal for QA abstention).

Compares the 5-signal classifier (existing 4 + score_entropy) against the
CURRENTLY DEPLOYED 4-signal classifier, both re-fit under the same fold
assignments, across the same 5 robustness seeds used to validate the
4-signal one -- so "beats the deployed baseline" means something real,
not just "beats a stale number in a file."

OFFLINE MEASUREMENT ONLY. Does not touch results/abstention_threshold_
newsignals.json (the deployed classifier) unless the result is a clear,
seed-robust improvement, matching this project's standing discipline.

Usage: python scripts/calibrate_abstention_entropy_signal.py
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
from scripts.calibrate_abstention_newsignals import compute_new_signals, ROBUSTNESS_SEEDS
from scripts.calibrate_abstention import SAMPLE_SEED, load_rows, build_synthetic_entity_calibration_set
from scripts.calibrate_abstention_kfold import make_folds, N_FOLDS

OUT_PATH = ROOT / "results" / "abstention_threshold_entropy_signal.json"
BASELINE_FEATURES = ["query_confidence", "query_top1_score", "question_match_any", "bm25_vector_agreement"]
NEW_FEATURES = BASELINE_FEATURES + ["score_entropy"]


def main():
    out_of_scope, answerable_sample = load_rows()
    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()
    print(f"Out-of-Scope: {len(out_of_scope)}, answerable: {len(answerable_sample)}, "
          f"synthetic: {len(synthetic_queries)}", flush=True)

    retriever = HybridRetriever()
    queries = out_of_scope + answerable_sample + synthetic_queries
    labels = [True] * len(out_of_scope) + [False] * len(answerable_sample) + synthetic_labels

    data = {"labels": [], "query_confidence": [], "query_top1_score": [],
            "question_match_any": [], "bm25_vector_agreement": [], "score_entropy": []}

    t0 = time.perf_counter()
    for i, (q, label) in enumerate(zip(queries, labels)):
        results, meta = retriever.retrieve_adaptive(q)
        if meta["route"] != "open_ended":
            continue
        if results:
            qc = results[0]["query_confidence"]
            qt = results[0]["query_top1_score"]
            se = results[0]["score_entropy"]
        else:
            qc, qt, se = 0.0, 0.0, 0.0
        qma, bva = compute_new_signals(retriever, q, results)
        data["labels"].append(label)
        data["query_confidence"].append(qc)
        data["query_top1_score"].append(qt)
        data["question_match_any"].append(qma)
        data["bm25_vector_agreement"].append(bva)
        data["score_entropy"].append(se)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} retrieved ({time.perf_counter()-t0:.1f}s)", flush=True)

    n = len(data["labels"])
    print(f"\nopen_ended n={n} (must match the deployed 4-signal classifier's n=371)", flush=True)
    print(f"score_entropy: mean={np.mean(data['score_entropy']):.4f} "
          f"std={np.std(data['score_entropy']):.4f} "
          f"range=[{min(data['score_entropy']):.4f}, {max(data['score_entropy']):.4f}]", flush=True)

    labels_arr = np.array(data["labels"], dtype=int)
    X_baseline = np.column_stack([np.array(data[f]) for f in BASELINE_FEATURES])
    X_new = np.column_stack([np.array(data[f]) for f in NEW_FEATURES])

    per_seed = []
    all_new_accs, all_baseline_accs = [], []
    for seed in ROBUSTNESS_SEEDS:
        folds = make_folds(data["labels"], N_FOLDS, seed)
        new_fold_accs, baseline_fold_accs = [], []
        for k in range(N_FOLDS):
            test_idx = folds[k]
            train_idx = [i for j, f in enumerate(folds) if j != k for i in f]

            clf_new = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf_new.fit(X_new[train_idx], labels_arr[train_idx])
            new_fold_accs.append(float((clf_new.predict(X_new[test_idx]) == labels_arr[test_idx]).mean()))

            clf_base = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf_base.fit(X_baseline[train_idx], labels_arr[train_idx])
            baseline_fold_accs.append(float((clf_base.predict(X_baseline[test_idx]) == labels_arr[test_idx]).mean()))

        seed_new_mean = float(np.mean(new_fold_accs))
        seed_baseline_mean = float(np.mean(baseline_fold_accs))
        per_seed.append({"seed": seed, "5signal_cv_accuracy": round(seed_new_mean, 4),
                          "4signal_cv_accuracy": round(seed_baseline_mean, 4)})
        all_new_accs.extend(new_fold_accs)
        all_baseline_accs.extend(baseline_fold_accs)
        print(f"  seed={seed}: 5-signal(+entropy)={seed_new_mean:.4f}  deployed-4-signal={seed_baseline_mean:.4f}", flush=True)

    new_mean, new_std = float(np.mean(all_new_accs)), float(np.std(all_new_accs))
    baseline_mean, baseline_std = float(np.mean(all_baseline_accs)), float(np.std(all_baseline_accs))
    n_wins = sum(1 for row in per_seed if row["5signal_cv_accuracy"] > row["4signal_cv_accuracy"])
    delta = new_mean - baseline_mean

    full_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    full_clf.fit(X_new, labels_arr)
    coefficients = dict(zip(NEW_FEATURES, [float(c) for c in full_clf.coef_[0]]))
    intercept = float(full_clf.intercept_[0])

    result = {
        "n": n, "n_folds": N_FOLDS, "robustness_seeds": ROBUSTNESS_SEEDS, "sample_seed": SAMPLE_SEED,
        "features": NEW_FEATURES, "per_seed": per_seed,
        "5signal_cv_mean_accuracy": round(new_mean, 4), "5signal_cv_std_accuracy": round(new_std, 4),
        "4signal_deployed_cv_mean_accuracy": round(baseline_mean, 4), "4signal_deployed_cv_std_accuracy": round(baseline_std, 4),
        "seeds_where_5signal_beat_4signal": f"{n_wins}/{len(ROBUSTNESS_SEEDS)}",
        "improvement_over_deployed_4signal": round(delta, 4),
        "full_data_coefficients": coefficients, "full_data_intercept": intercept,
    }

    print(f"\nAcross {len(ROBUSTNESS_SEEDS)} seeds x {N_FOLDS} folds ({len(all_new_accs)} total test folds):")
    print(f"  5-signal (+score_entropy): mean={new_mean:.4f} std={new_std:.4f}")
    print(f"  4-signal (currently deployed): mean={baseline_mean:.4f} std={baseline_std:.4f}")
    print(f"  5-signal beat deployed 4-signal in {n_wins}/{len(ROBUSTNESS_SEEDS)} seeds, mean delta {delta:+.4f}")
    if n_wins == len(ROBUSTNESS_SEEDS) and delta > 0.02:
        print("  Consistent, non-trivial further win -- worth deploying.")
    elif delta > 0.02 and n_wins >= len(ROBUSTNESS_SEEDS) - 1:
        print("  Real but not unanimous improvement -- borderline.")
    else:
        print("  Not a reliable further improvement. NOT deploying score_entropy. "
              "Honest negative result: this specific 5th signal does not help beyond the 4 already deployed.")
    print(f"Full-data coefficients: {coefficients}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
