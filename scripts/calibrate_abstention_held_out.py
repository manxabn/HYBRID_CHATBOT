"""
Held-out re-validation of the abstention threshold, closing a real
methodological gap found this session: scripts/calibrate_abstention.py
sweeps the threshold to maximize accuracy on the SAME set it then reports
accuracy on -- there is no train/test split anywhere in it. The reported
entity_heavy accuracy (0.95, n=60) and open_ended accuracy (0.710, n=372)
in results/abstention_threshold.json are therefore in-sample fit quality,
not a genuine generalization estimate. Not a false claim in the paper
(Section subsec:abstention only ever says "calibration", never "held-out"
or "test set"), but an undisclosed limitation worth actually measuring
rather than just flagging.

Method: reuses calibrate_abstention.py's exact data-loading, signal
-sweeping, and metric functions unmodified (import, not reimplementation
-- avoids the "two copies of the same logic silently drift apart" bug
class this project has hit before, e.g. patterns.py's own docstring).
For each route, stratified train/test split by label (70/30, seed=42,
same convention as finetune_embeddings.py's split): sweep the threshold
for max accuracy on TRAIN only, then report precision/recall/f1/accuracy
on TEST only (never seen during threshold selection) -- this is the
genuine generalization estimate the original script never computed.

Usage: python scripts/calibrate_abstention_held_out.py
"""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from scripts.calibrate_abstention import (
    SIGNALS, SAMPLE_SEED, load_rows, build_synthetic_entity_calibration_set,
    precision_recall_f1, clopper_pearson_ci,
)

OUT_PATH = ROOT / "results" / "abstention_threshold_held_out.json"
TRAIN_FRACTION = 0.7
SPLIT_SEED = 42


def stratified_split(labels, fraction=TRAIN_FRACTION, seed=SPLIT_SEED):
    """Returns (train_idx, test_idx), stratified by label so both splits
    keep roughly the same should-abstain/answerable balance as the whole
    set -- avoids a degenerate split where one side is almost all one class."""
    rng = random.Random(seed)
    pos_idx = [i for i, l in enumerate(labels) if l]
    neg_idx = [i for i, l in enumerate(labels) if not l]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_train = int(round(len(pos_idx) * fraction))
    n_neg_train = int(round(len(neg_idx) * fraction))
    train_idx = set(pos_idx[:n_pos_train] + neg_idx[:n_neg_train])
    test_idx = set(range(len(labels))) - train_idx
    return sorted(train_idx), sorted(test_idx)


def sweep_signal_on_indices(labels, confidences, indices):
    """Same accuracy-maximizing sweep as calibrate_abstention.py's
    sweep_signal(), but restricted to a subset of indices (the train
    split) -- returns the best (threshold, signal-name-agnostic) row."""
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
    print(f"Out-of-Scope: {len(out_of_scope)}, answerable sample: {len(answerable_sample)}, "
          f"synthetic entity-heavy: {len(synthetic_queries)}", flush=True)

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
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(queries)} retrieved ({time.perf_counter()-t0:.1f}s)", flush=True)

    result = {"train_fraction": TRAIN_FRACTION, "split_seed": SPLIT_SEED, "sample_seed": SAMPLE_SEED}
    for route_name, data in by_route.items():
        n = len(data["labels"])
        if n == 0:
            print(f"WARNING: 0 queries routed to {route_name} -- skipping")
            continue
        train_idx, test_idx = stratified_split(data["labels"])
        print(f"\n[{route_name}] n={n}, train={len(train_idx)}, test={len(test_idx)}")

        # Pick the best signal ON TRAIN ONLY (the original script picks the
        # best signal using the SAME full-set accuracy it then reports --
        # here signal selection is also confined to train, so test is
        # untouched until final evaluation).
        route_best_train = None
        for signal_name in SIGNALS:
            best = sweep_signal_on_indices(data["labels"], data[signal_name], train_idx)
            if route_best_train is None or best["accuracy"] > route_best_train["accuracy"]:
                route_best_train = {**best, "signal": signal_name}

        # Evaluate that exact threshold on TEST -- never used for selection.
        test_labels = [data["labels"][i] for i in test_idx]
        test_conf = [data[route_best_train["signal"]][i] for i in test_idx]
        test_metrics = precision_recall_f1(test_labels, test_conf, route_best_train["threshold"])
        n_correct = test_metrics["tp"] + test_metrics["tn"]
        acc_ci_lo, acc_ci_hi = clopper_pearson_ci(n_correct, len(test_idx))

        result[route_name] = {
            "signal": route_best_train["signal"], "threshold": route_best_train["threshold"],
            "train_accuracy": round(route_best_train["accuracy"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "n_train": len(train_idx), "n_test": len(test_idx),
            "test_accuracy_ci95_lo": round(acc_ci_lo, 4), "test_accuracy_ci95_hi": round(acc_ci_hi, 4),
        }
        overfit_gap = route_best_train["accuracy"] - test_metrics["accuracy"]
        print(f"  signal={route_best_train['signal']} threshold={route_best_train['threshold']:.4f}")
        print(f"  train_accuracy={route_best_train['accuracy']:.4f}  "
              f"test_accuracy={test_metrics['accuracy']:.4f}  "
              f"(95% CI [{acc_ci_lo:.4f}, {acc_ci_hi:.4f}])  "
              f"overfit_gap={overfit_gap:+.4f}")
        if len(test_idx) < 30:
            print(f"  WARNING: test n={len(test_idx)} is small -- CI is wide by construction.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
