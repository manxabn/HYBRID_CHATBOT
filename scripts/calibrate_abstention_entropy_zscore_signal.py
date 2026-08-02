"""
A genuine fix attempt on the score_entropy negative result (calibrate_
abstention_entropy_signal.py: 3/5 seeds, delta +0.0011, noise), not a
different signal -- a diagnosis of WHY it failed and a direct fix for that
specific cause.

Root cause (visible in the prior script's own output): score_entropy had
almost no variance on the open-ended calibration set (mean 0.988, std
0.005, range [0.968, 0.999]) -- nearly maxed out (near-uniform) for almost
every query. This corpus's linear-fusion scores for the open-ended route
sit in a narrow absolute band (typically ~0.7-0.9) regardless of how
confidently separated the candidates actually are -- softmax over values
that close together always looks close to uniform, no matter the TRUE
relative spread between them. That is a scale artifact of raw softmax on
narrow-range inputs, not evidence that "how spread out are the retrieval
scores" is an uninformative question for this corpus.

Fix (pipeline/hybrid_retriever.py, 2026-08-03): standardize (z-score) the
candidate pool's scores before the softmax -- score_entropy_zscore. This
is parameter-free (no temperature or other hyperparameter to tune on the
same data used for evaluation, which would repeat the in-sample-overfit
mistake this project has already caught and fixed elsewhere) and removes
the absolute-scale dependence while preserving the RELATIVE shape of the
distribution. On a handful of real spot-check queries this already shows
much more spread (0.32-0.89) than the original (0.00-0.998 but clustered
at the high end for nearly every query) -- worth testing properly before
concluding anything.

Same seed-robustness protocol as every other abstention-signal
experiment in this project: compares a 5-signal classifier (deployed 4 +
score_entropy_zscore) against the currently-deployed 4-signal one, 5
independent fold-assignment seeds x 5 folds. OFFLINE MEASUREMENT ONLY.

HISTORICAL RECORD, NOT RE-RUNNABLE AS-IS (2026-08-03): despite this
script's more promising result than the plain entropy version (4/5 seeds,
delta +0.0102 vs. +0.0011), a paired t-test/Wilcoxon check (added to this
same script) found the two tests disagreed (p=0.147, p=0.052) -- not
enough to clear this project's "both tests must agree" deployment bar, so
score_entropy_zscore was removed from hybrid_retriever.py's retrieve()
alongside the plain version. Re-running this script today will KeyError
on results[0]["score_entropy_zscore"]; results/abstention_threshold_
entropy_zscore_signal.json is the permanent record of what was measured.

Usage (historical): python scripts/calibrate_abstention_entropy_zscore_signal.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from scripts.calibrate_abstention_newsignals import compute_new_signals, ROBUSTNESS_SEEDS
from scripts.calibrate_abstention import SAMPLE_SEED, load_rows, build_synthetic_entity_calibration_set
from scripts.calibrate_abstention_kfold import make_folds, N_FOLDS

OUT_PATH = ROOT / "results" / "abstention_threshold_entropy_zscore_signal.json"
BASELINE_FEATURES = ["query_confidence", "query_top1_score", "question_match_any", "bm25_vector_agreement"]
NEW_FEATURES = BASELINE_FEATURES + ["score_entropy_zscore"]


def main():
    out_of_scope, answerable_sample = load_rows()
    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()
    print(f"Out-of-Scope: {len(out_of_scope)}, answerable: {len(answerable_sample)}, "
          f"synthetic: {len(synthetic_queries)}", flush=True)

    retriever = HybridRetriever()
    queries = out_of_scope + answerable_sample + synthetic_queries
    labels = [True] * len(out_of_scope) + [False] * len(answerable_sample) + synthetic_labels

    data = {"labels": [], "query_confidence": [], "query_top1_score": [],
            "question_match_any": [], "bm25_vector_agreement": [], "score_entropy_zscore": []}

    t0 = time.perf_counter()
    for i, (q, label) in enumerate(zip(queries, labels)):
        results, meta = retriever.retrieve_adaptive(q)
        if meta["route"] != "open_ended":
            continue
        if results:
            qc = results[0]["query_confidence"]
            qt = results[0]["query_top1_score"]
            sez = results[0]["score_entropy_zscore"]
        else:
            qc, qt, sez = 0.0, 0.0, 0.0
        qma, bva = compute_new_signals(retriever, q, results)
        data["labels"].append(label)
        data["query_confidence"].append(qc)
        data["query_top1_score"].append(qt)
        data["question_match_any"].append(qma)
        data["bm25_vector_agreement"].append(bva)
        data["score_entropy_zscore"].append(sez)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} retrieved ({time.perf_counter()-t0:.1f}s)", flush=True)

    n = len(data["labels"])
    print(f"\nopen_ended n={n} (must match the deployed 4-signal classifier's n=371)", flush=True)
    print(f"score_entropy_zscore: mean={np.mean(data['score_entropy_zscore']):.4f} "
          f"std={np.std(data['score_entropy_zscore']):.4f} "
          f"range=[{min(data['score_entropy_zscore']):.4f}, {max(data['score_entropy_zscore']):.4f}]", flush=True)

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
        print(f"  seed={seed}: 5-signal(+entropy_zscore)={seed_new_mean:.4f}  deployed-4-signal={seed_baseline_mean:.4f}", flush=True)

    new_mean, new_std = float(np.mean(all_new_accs)), float(np.std(all_new_accs))
    baseline_mean, baseline_std = float(np.mean(all_baseline_accs)), float(np.std(all_baseline_accs))
    n_wins = sum(1 for row in per_seed if row["5signal_cv_accuracy"] > row["4signal_cv_accuracy"])
    delta = new_mean - baseline_mean

    # Paired significance test (this project's standing convention for
    # comparing two configurations on the SAME evaluation units, e.g.
    # measure_ir_metrics.py's McNemar+bootstrap and compute_faithfulness's
    # paired t-test+Wilcoxon): all_new_accs[i]/all_baseline_accs[i] are the
    # SAME (seed, fold) test split for both classifiers, so this is a
    # genuine paired comparison, not two independent samples. "Both tests
    # must agree" before treating a result as more than suggestive.
    paired_diffs = np.array(all_new_accs) - np.array(all_baseline_accs)
    t_stat, t_p = stats.ttest_rel(all_new_accs, all_baseline_accs)
    try:
        w_stat, w_p = stats.wilcoxon(all_new_accs, all_baseline_accs)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0  # all differences zero, or too few non-zero pairs

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
        "paired_ttest_p": round(float(t_p), 4),
        "paired_wilcoxon_p": round(float(w_p), 4),
        "full_data_coefficients": coefficients, "full_data_intercept": intercept,
    }

    print(f"\nAcross {len(ROBUSTNESS_SEEDS)} seeds x {N_FOLDS} folds ({len(all_new_accs)} total test folds, paired):")
    print(f"  5-signal (+score_entropy_zscore): mean={new_mean:.4f} std={new_std:.4f}")
    print(f"  4-signal (currently deployed): mean={baseline_mean:.4f} std={baseline_std:.4f}")
    print(f"  5-signal beat deployed 4-signal in {n_wins}/{len(ROBUSTNESS_SEEDS)} seeds, mean delta {delta:+.4f}")
    print(f"  Paired t-test: p={t_p:.4f}   Paired Wilcoxon signed-rank: p={w_p:.4f}")
    if t_p < 0.05 and w_p < 0.05:
        print("  BOTH paired tests agree this is significant -- a real, deployable improvement.")
    elif t_p < 0.05 or w_p < 0.05:
        print("  Tests DISAGREE (only one significant) -- inconclusive by this project's own "
              "standing convention (both tests must agree). Treating as not yet deployable.")
    else:
        print("  Neither test significant -- not deploying score_entropy_zscore. A real, "
              "diagnosed, directionally-consistent (4/5 seeds) improvement, but too small "
              f"relative to fold-to-fold noise (n={len(all_new_accs)} paired folds) to trust "
              "as more than suggestive.")
    print(f"Full-data coefficients: {coefficients}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
