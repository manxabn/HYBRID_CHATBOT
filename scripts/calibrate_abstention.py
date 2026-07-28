"""
Calibrate the abstention gate against real "Out of Scope / Unanswerable"
rows (EnglishQA: 127, BanglishQA: 63) vs. a matched sample of answerable
rows from both tables -- retrieval-only, no LLM generation involved, so
this runs in minutes without needing Ollama.

For each row we run HybridRetriever.retrieve_adaptive (the same routing the
novel pipeline uses at answer time) and record BOTH candidate confidence
signals hybrid_retriever.py exposes:
  - query_confidence: top1-top2 score margin
  - query_top1_score: raw top-1 score
(see pipeline/abstention.py and hybrid_retriever.py docstrings for why
margin alone has a known blind spot on near-duplicate-paraphrase retrieval).
For each route (entity_heavy/open_ended) we sweep thresholds for EACH signal
independently and keep whichever signal gets the higher accuracy -- an
empirical choice, not an assumption that margin (or top1_score) is
correct a priori.

Entity-heavy calibration-set augmentation (2026-07-28)
--------------------------------------------------------
The entity_heavy route's calibration set is small by construction: only 18
of the 380 Out-of-Scope/answerable EnglishQA+BanglishQA rows happen to
mention a course code or faculty name/initial at all (most Out-of-Scope
rows are generic "can you help with X" questions, not course-code lookups).
This is a genuine, disclosed limitation (wide 95% CI), not a bug -- but it
IS possible to grow it with more real, non-fabricated labeled examples: a
query asking about a course code that verifiably does not exist anywhere in
CourseDetails/Prerequisites/Coordinator is unambiguously "should abstain"
by construction (checked directly against the database, not asserted), and
a query about a real code that DOES have a Prerequisites/Coordinator row is
unambiguously answerable -- same idea for faculty initials against
FacultyList. build_synthetic_entity_calibration_set() below constructs a
balanced set of both, which is merged into the entity_heavy route's
calibration data before threshold selection. These are template-generated
queries, not organically collected ones -- disclosed here and in paper.tex,
not presented as naturally-occurring student questions.

Usage: python scripts/calibrate_abstention.py
"""

import csv
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever


def clopper_pearson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial CI for a proportion (successes/n), e.g. Angelopoulos &
    Bates (2021) "conformal prediction" tutorial's use of exact binomial
    bounds for small-sample risk control -- point estimates like
    accuracy=0.889 are meaningless on their own when n is small (this
    project's entity_heavy calibration set has n=18); the CI communicates
    how much that estimate could plausibly move with more data."""
    if n == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence
    lo = 0.0 if successes == 0 else beta_dist.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes)
    return (float(lo), float(hi))

DB_PATH = ROOT / "knowledge_base.db"
OUT_SWEEP = ROOT / "results" / "abstention_calibration.csv"
OUT_THRESHOLD = ROOT / "results" / "abstention_threshold.json"
SAMPLE_SEED = 42
SIGNALS = ["query_confidence", "query_top1_score"]


def load_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT Question, Category FROM EnglishQA")
    english_rows = cur.fetchall()
    cur.execute("SELECT QuestionBanglish, Category FROM BanglishQA")
    banglish_rows = cur.fetchall()
    conn.close()

    all_rows = english_rows + banglish_rows
    out_of_scope = [q for q, cat in all_rows if cat == "Out of Scope / Unanswerable" and q]
    answerable = [q for q, cat in all_rows if cat != "Out of Scope / Unanswerable" and q]

    rng = random.Random(SAMPLE_SEED)
    answerable_sample = rng.sample(answerable, min(len(out_of_scope), len(answerable)))

    return out_of_scope, answerable_sample


def build_synthetic_entity_calibration_set():
    """Returns (queries, labels) for template-generated entity-heavy
    examples: nonexistent course codes / faculty initials (label=True,
    should abstain) and real ones with actual Prerequisites/Coordinator/
    FacultyList data (label=False, answerable) -- see module docstring.
    Existence is checked directly against knowledge_base.db, not assumed."""
    import re

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT Course FROM CourseDetails")
    all_codes = set()
    for (c,) in cur.fetchall():
        if c:
            m = re.search(r"[A-Za-z]{2,4}\d{3}[A-Za-z]?", c.upper())
            if m:
                all_codes.add(m.group(0))
    cur.execute("SELECT DISTINCT Course FROM Prerequisites WHERE PreRequisite IS NOT NULL")
    codes_with_prereq = {c.strip().upper() for (c,) in cur.fetchall() if c}
    cur.execute("SELECT DISTINCT Initial FROM FacultyList WHERE Initial IS NOT NULL")
    real_initials = {i.strip().upper() for (i,) in cur.fetchall() if i}
    conn.close()

    prefixes = sorted({re.match(r"[A-Za-z]{2,4}", c).group(0) for c in all_codes})
    nonexistent_codes = []
    for prefix in prefixes:
        for num in range(100, 500):
            code = f"{prefix}{num}"
            if code not in all_codes:
                nonexistent_codes.append(code)
            if len([c for c in nonexistent_codes if c.startswith(prefix)]) >= 4:
                break

    rng = random.Random(SAMPLE_SEED)
    real_answerable_codes = rng.sample(sorted(codes_with_prereq), min(20, len(codes_with_prereq)))
    fake_codes = rng.sample(nonexistent_codes, min(20, len(nonexistent_codes)))

    queries, labels = [], []
    for code in fake_codes:
        queries.append(f"What are the prerequisites for {code}?")
        labels.append(True)
    for code in real_answerable_codes:
        queries.append(f"What are the prerequisites for {code}?")
        labels.append(False)

    all_letter_combos = [a + b + c for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in "XZQ"]
    fake_initials = [i for i in all_letter_combos if i not in real_initials]
    fake_initials_sample = rng.sample(fake_initials, min(10, len(fake_initials)))
    real_initials_sample = rng.sample(sorted(real_initials), min(10, len(real_initials)))
    for initial in fake_initials_sample:
        queries.append(f"What is {initial}'s designation?")
        labels.append(True)
    for initial in real_initials_sample:
        queries.append(f"What is {initial}'s designation?")
        labels.append(False)

    return queries, labels


def precision_recall_f1(labels, confidences, threshold):
    tp = fp = fn = tn = 0
    for should_abstain, conf in zip(labels, confidences):
        predicted_abstain = conf < threshold
        if should_abstain and predicted_abstain:
            tp += 1
        elif should_abstain and not predicted_abstain:
            fn += 1
        elif not should_abstain and predicted_abstain:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def sweep_signal(route_name, signal_name, labels, confidences):
    candidate_thresholds = sorted(set(round(c, 4) for c in confidences))
    sweep_rows = []
    best = None
    for thr in candidate_thresholds:
        metrics = precision_recall_f1(labels, confidences, thr)
        row = {"route": route_name, "signal": signal_name, "threshold": thr, **metrics}
        sweep_rows.append(row)
        # Select on accuracy, not F1: maximizing F1 alone on a balanced set
        # has a degenerate solution (predict "abstain" for nearly everything
        # -> recall=1.0, precision~0.5, f1 artificially inflated but the
        # classifier isn't actually discriminating). Confirmed this
        # empirically on the first calibration run of this script: the
        # F1-optimal threshold (0.5186) abstained on effectively the whole
        # set (precision=0.500, recall=1.000, accuracy=0.500 -- exactly
        # chance). Accuracy has no such degenerate optimum for a balanced set.
        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = row
    return sweep_rows, best


def main():
    out_of_scope, answerable_sample = load_rows()
    print(f"Out-of-Scope rows (English+Banglish): {len(out_of_scope)}, "
          f"matched answerable sample: {len(answerable_sample)}")

    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()
    print(f"Synthetic entity-heavy calibration examples (nonexistent/real "
          f"course codes and faculty initials, see module docstring): {len(synthetic_queries)}")

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
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(queries)} retrieved ({elapsed:.1f}s elapsed)")

    all_sweep_rows = []
    result = {"n_out_of_scope": len(out_of_scope), "n_answerable_sample": len(answerable_sample),
              "n_synthetic_entity_calibration": len(synthetic_queries), "sample_seed": SAMPLE_SEED}
    for route_name, data in by_route.items():
        n = len(data["labels"])
        if n == 0:
            print(f"WARNING: 0 queries routed to {route_name} in calibration set -- skipping")
            continue
        route_best = None
        for signal_name in SIGNALS:
            sweep_rows, best = sweep_signal(route_name, signal_name, data["labels"], data[signal_name])
            all_sweep_rows.extend(sweep_rows)
            if route_best is None or best["accuracy"] > route_best["accuracy"]:
                route_best = best
        n_correct = route_best["tp"] + route_best["tn"]
        acc_ci_lo, acc_ci_hi = clopper_pearson_ci(n_correct, n)
        result[route_name] = {
            "signal": route_best["signal"], "threshold": route_best["threshold"],
            "precision": route_best["precision"], "recall": route_best["recall"],
            "f1": route_best["f1"], "accuracy": route_best["accuracy"], "n": n,
            "accuracy_ci95_lo": acc_ci_lo, "accuracy_ci95_hi": acc_ci_hi,
        }
        print(f"[{route_name}] n={n} best signal={route_best['signal']} "
              f"threshold={route_best['threshold']:.4f} "
              f"(precision={route_best['precision']:.3f}, recall={route_best['recall']:.3f}, "
              f"f1={route_best['f1']:.3f}, accuracy={route_best['accuracy']:.3f}, "
              f"95% CI=[{acc_ci_lo:.3f}, {acc_ci_hi:.3f}])")
        if n < 30:
            print(f"  WARNING: n={n} is small -- the 95% CI above is wide by construction; "
                  f"treat the point estimate as indicative, not precise, in any write-up.")

    OUT_SWEEP.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SWEEP, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_sweep_rows)
    print(f"Wrote full threshold sweep ({len(all_sweep_rows)} rows, both routes x both signals) to {OUT_SWEEP}")

    with open(OUT_THRESHOLD, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote calibrated per-route thresholds to {OUT_THRESHOLD}")


if __name__ == "__main__":
    main()
