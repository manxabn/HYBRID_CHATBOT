"""
The actual re-calibration this project's held-out check (calibrate_
abstention_held_out.py) motivated but didn't itself produce: that script
measured the open-ended route's generalization gap (train 0.673 -> test
0.532 on one 70/30 split) but a single split's chosen threshold is
sensitive to which points happened to land in train vs.\ test -- there is
no principled way to prefer that one split's threshold over a different
split's. Refitting on 100% of the data with the same max-accuracy-sweep
method (what the ORIGINAL calibrate_abstention.py already does) would
just reproduce the same in-sample-overfit number, not fix it -- more data
does not fix a systematically-biased selection procedure. A genuinely
new signal was already tried and ruled out earlier this project
(cross-validated 3-signal logistic regression, 74.7% CV accuracy vs.\
73.3% majority baseline -- not worth deploying), so this does not attempt
another one.

What this DOES do, a real methodological improvement over both the
original (no split) and the single-split held-out check: 5-fold cross
-validation. For each fold, sweep the threshold on the other 4 folds
(train) and evaluate on the held-out fold (test) -- averaging accuracy
across all 5 folds gives a threshold-selection-robust generalization
estimate that doesn't depend on one arbitrary split, and the final
deployed threshold is chosen by the SAME max-accuracy-on-train logic but
now cross-validated 5 times rather than validated once, which is the
standard way to make small-n threshold selection more stable without
inventing new data or a new signal.

Usage: python scripts/calibrate_abstention_kfold.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from scripts.calibrate_abstention import (
    SIGNALS, SAMPLE_SEED, load_rows, build_synthetic_entity_calibration_set,
    precision_recall_f1, clopper_pearson_ci,
)

OUT_PATH = ROOT / "results" / "abstention_threshold_kfold.json"
N_FOLDS = 5
SPLIT_SEED = 42


def make_folds(labels, n_folds=N_FOLDS, seed=SPLIT_SEED):
    """Stratified k-fold indices, same balance logic as the held-out
    script's stratified split, extended to n_folds groups instead of 2."""
    rng = random.Random(seed)
    pos_idx = [i for i, l in enumerate(labels) if l]
    neg_idx = [i for i, l in enumerate(labels) if not l]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos_idx):
        folds[i % n_folds].append(idx)
    for i, idx in enumerate(neg_idx):
        folds[i % n_folds].append(idx)
    return [sorted(f) for f in folds]


def sweep_best(labels, confidences, indices):
    sub_labels = [labels[i] for i in indices]
    sub_conf = [confidences[i] for i in indices]
    candidate_thresholds = sorted(set(round(c, 4) for c in sub_conf))
    best = None
    for thr in candidate_thresholds:
        metrics = precision_recall_f1(sub_labels, sub_conf, thr)
        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = {"threshold": thr, **metrics}
    return best


def main():
    out_of_scope, answerable_sample = load_rows()
    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()
    print(f"Out-of-Scope: {len(out_of_scope)}, answerable: {len(answerable_sample)}, "
          f"synthetic: {len(synthetic_queries)}", flush=True)

    retriever = HybridRetriever()
    queries = out_of_scope + answerable_sample + synthetic_queries
    labels = [True] * len(out_of_scope) + [False] * len(answerable_sample) + synthetic_labels

    by_route = {"entity_heavy": {"labels": [], "query_confidence": [], "query_top1_score": []},
                "open_ended": {"labels": [], "query_confidence": [], "query_top1_score": []}}

    t0 = time.perf_counter()
    for i, (q, label) in enumerate(zip(queries, labels)):
        results, meta = retriever.retrieve_adaptive(q)
        route = meta["route"]
        if results:
            by_route[route]["query_confidence"].append(results[0]["query_confidence"])
            by_route[route]["query_top1_score"].append(results[0]["query_top1_score"])
        else:
            by_route[route]["query_confidence"].append(0.0)
            by_route[route]["query_top1_score"].append(0.0)
        by_route[route]["labels"].append(label)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} retrieved ({time.perf_counter()-t0:.1f}s)", flush=True)

    result = {"n_folds": N_FOLDS, "split_seed": SPLIT_SEED, "sample_seed": SAMPLE_SEED}
    for route_name, data in by_route.items():
        n = len(data["labels"])
        if n == 0:
            continue
        folds = make_folds(data["labels"], N_FOLDS)
        fold_accuracies = []
        fold_thresholds = []
        fold_signals = []
        for k in range(N_FOLDS):
            test_idx = folds[k]
            train_idx = [i for j, f in enumerate(folds) if j != k for i in f]
            route_best_train = None
            for signal_name in SIGNALS:
                best = sweep_best(data["labels"], data[signal_name], train_idx)
                if route_best_train is None or best["accuracy"] > route_best_train["accuracy"]:
                    route_best_train = {**best, "signal": signal_name}
            test_labels = [data["labels"][i] for i in test_idx]
            test_conf = [data[route_best_train["signal"]][i] for i in test_idx]
            test_metrics = precision_recall_f1(test_labels, test_conf, route_best_train["threshold"])
            fold_accuracies.append(test_metrics["accuracy"])
            fold_thresholds.append(route_best_train["threshold"])
            fold_signals.append(route_best_train["signal"])
            print(f"  [{route_name}] fold {k+1}/{N_FOLDS}: signal={route_best_train['signal']} "
                  f"threshold={route_best_train['threshold']:.4f} test_acc={test_metrics['accuracy']:.4f}", flush=True)

        cv_mean_acc = float(np.mean(fold_accuracies))
        cv_std_acc = float(np.std(fold_accuracies))
        # Final deployed signal: whichever signal won the majority of folds
        # (robust choice, not just the last fold's). Final threshold: the
        # median of the per-fold thresholds for that signal (robust to a
        # single fold's outlier threshold, standard k-fold-to-deployment
        # aggregation choice).
        from collections import Counter
        winning_signal = Counter(fold_signals).most_common(1)[0][0]
        matching_thresholds = [t for t, s in zip(fold_thresholds, fold_signals) if s == winning_signal]
        final_threshold = float(np.median(matching_thresholds))

        final_metrics = precision_recall_f1(data["labels"], data[winning_signal], final_threshold)
        n_correct = final_metrics["tp"] + final_metrics["tn"]
        acc_ci_lo, acc_ci_hi = clopper_pearson_ci(n_correct, n)

        result[route_name] = {
            "signal": winning_signal, "threshold": final_threshold,
            "cv_mean_accuracy": round(cv_mean_acc, 4), "cv_std_accuracy": round(cv_std_acc, 4),
            "fold_accuracies": [round(a, 4) for a in fold_accuracies],
            "full_data_accuracy_at_final_threshold": round(final_metrics["accuracy"], 4),
            "full_data_accuracy_ci95_lo": round(acc_ci_lo, 4), "full_data_accuracy_ci95_hi": round(acc_ci_hi, 4),
            "n": n,
        }
        print(f"[{route_name}] CV mean accuracy={cv_mean_acc:.4f} (std={cv_std_acc:.4f}) -- "
              f"this is the honest generalization estimate. Final deployed threshold={final_threshold:.4f} "
              f"(signal={winning_signal}, median across folds where it won)", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
