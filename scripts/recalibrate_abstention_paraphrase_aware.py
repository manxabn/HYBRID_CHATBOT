"""
Real fix attempt #2 for the paraphrase-robustness gap (results/paraphrase_
robustness_raw.csv: 20/20 open-ended paraphrases incorrectly abstained).
Fix attempt #1 (a semantic s_vec override, scripts/test_semantic_
sufficiency_override.py) was tested honestly and failed -- out-of-scope
queries scored HIGHER average semantic similarity than the paraphrases
needing rescue, so no threshold helped more than it harmed.

This is a different, more principled approach: instead of adding a new
heuristic signal, re-run the SAME calibration procedure as scripts/
calibrate_abstention.py, but fold real paraphrased-but-answerable queries
into the open_ended route's "answerable" calibration sample, so the
threshold itself is chosen against a training distribution that actually
includes paraphrased phrasing -- not just original wording, which is what
the deployed threshold was silently calibrated against.

Held-out discipline (same pattern as scripts/eval_lambda_held_out.py):
the 10 open-ended base queries in data/paraphrase_robustness_queries.csv
are split 5-tuning / 5-held-out by base_query_id. ONLY the 5 tuning bases'
paraphrases (10 rows: 2 paraphrases x 5 bases) are added to calibration.
The 5 held-out bases' paraphrases (10 rows) are used ONLY afterward, to
check whether the new threshold generalizes to paraphrases never used to
pick it -- exactly the check that caught the lambda=0.3 tuning-set peak
as noise earlier this session, applied here to guard against the same
overfitting risk.

Also re-checks the true-negative (out-of-scope) rate on a FRESH sample of
real out-of-scope queries not used in either the original or this
augmented calibration, since a threshold that rescues paraphrases by
becoming too lenient would trade a known problem for a worse, less
visible one.

Usage: python scripts/recalibrate_abstention_paraphrase_aware.py
"""

import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from scripts.calibrate_abstention import (
    load_rows, build_synthetic_entity_calibration_set, sweep_signal,
    clopper_pearson_ci, SAMPLE_SEED,
)

PARAPHRASE_QUERIES_PATH = ROOT / "data" / "paraphrase_robustness_queries.csv"
OUT_THRESHOLD = ROOT / "results" / "abstention_threshold_paraphrase_aware.json"
OUT_HELDOUT_CHECK = ROOT / "results" / "paraphrase_recalibration_heldout_check.csv"

TUNING_BASES = {"Q023", "Q027", "Q038", "Q016", "Q050"}
HELDOUT_BASES = {"Q087", "Q072", "Q079", "Q094", "Q092"}
FRESH_OOS_SEED = 4242
N_FRESH_OOS = 20


def load_fresh_out_of_scope(exclude_seed_sample_size, n):
    """A held-out out-of-scope sample, disjoint (different RNG seed, and
    checked for overlap) from the one used in the original/this
    calibration, to verify the new threshold's true-negative rate on
    queries it has never seen in any form."""
    import sqlite3
    conn = sqlite3.connect(ROOT / "knowledge_base.db")
    cur = conn.cursor()
    cur.execute("SELECT Question, Category FROM EnglishQA")
    rows = cur.fetchall()
    conn.close()
    oos = [q for q, cat in rows if cat == "Out of Scope / Unanswerable" and q]
    original_sample = set(random.Random(SAMPLE_SEED).sample(oos, min(exclude_seed_sample_size, len(oos))))
    remaining = [q for q in oos if q not in original_sample]
    return random.Random(FRESH_OOS_SEED).sample(remaining, min(n, len(remaining)))


def main():
    out_of_scope, answerable_sample = load_rows()
    synthetic_queries, synthetic_labels = build_synthetic_entity_calibration_set()

    para_df = pd.read_csv(PARAPHRASE_QUERIES_PATH)
    tuning_rows = para_df[(para_df["is_entity_heavy"] == False) &
                           (para_df["base_query_id"].isin(TUNING_BASES)) &
                           (para_df["variant"] != "original")]
    heldout_rows = para_df[(para_df["is_entity_heavy"] == False) &
                            (para_df["base_query_id"].isin(HELDOUT_BASES)) &
                            (para_df["variant"] != "original")]
    print(f"Tuning paraphrases added to calibration: {len(tuning_rows)}", flush=True)
    print(f"Held-out paraphrases (validation only, never in calibration): {len(heldout_rows)}", flush=True)

    tuning_queries = list(tuning_rows["query"])
    tuning_labels = [False] * len(tuning_queries)  # False = answerable, should NOT abstain

    queries = out_of_scope + answerable_sample + synthetic_queries + tuning_queries
    labels = ([True] * len(out_of_scope) + [False] * len(answerable_sample) +
               synthetic_labels + tuning_labels)

    retriever = HybridRetriever()

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

    result = {"n_out_of_scope": len(out_of_scope), "n_answerable_sample": len(answerable_sample),
              "n_synthetic_entity_calibration": len(synthetic_queries),
              "n_tuning_paraphrases_added": len(tuning_queries), "sample_seed": SAMPLE_SEED}
    for route_name, data in by_route.items():
        n = len(data["labels"])
        if n == 0:
            continue
        route_best = None
        for signal_name in ["query_confidence", "query_top1_score"]:
            _, best = sweep_signal(route_name, signal_name, data["labels"], data[signal_name])
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
        print(f"[{route_name}] signal={route_best['signal']} threshold={route_best['threshold']} "
              f"accuracy={route_best['accuracy']:.3f} (n={n})", flush=True)

    import json
    with open(OUT_THRESHOLD, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_THRESHOLD}", flush=True)

    # --- Validation: does the NEW open_ended threshold rescue the HELD-OUT
    # paraphrases (never used to pick it), and does it preserve the true
    # -abstention rate on a FRESH out-of-scope sample? ---
    new_signal = result["open_ended"]["signal"]
    new_threshold = result["open_ended"]["threshold"]
    print(f"\n=== Validating open_ended: signal={new_signal}, threshold={new_threshold} ===", flush=True)

    rows = []
    for _, r in heldout_rows.iterrows():
        results, meta = retriever.retrieve_adaptive(r["query"])
        sig_val = results[0][new_signal] if results else 0.0
        would_abstain = sig_val < new_threshold
        rows.append({"group": "heldout_paraphrase_SHOULD_NOT_abstain", "query_id": r["query_id"],
                     "query": r["query"], "signal_value": sig_val, "would_abstain_new_threshold": would_abstain})

    fresh_oos = load_fresh_out_of_scope(len(out_of_scope), N_FRESH_OOS)
    for i, q in enumerate(fresh_oos):
        results, meta = retriever.retrieve_adaptive(q)
        if meta["route"] != "open_ended":
            continue
        sig_val = results[0][new_signal] if results else 0.0
        would_abstain = sig_val < new_threshold
        rows.append({"group": "fresh_out_of_scope_SHOULD_abstain", "query_id": f"FRESH-OOS-{i}",
                     "query": q, "signal_value": sig_val, "would_abstain_new_threshold": would_abstain})

    check_df = pd.DataFrame(rows)
    check_df.to_csv(OUT_HELDOUT_CHECK, index=False)

    rescued = check_df[check_df["group"].str.contains("SHOULD_NOT")]
    still_abstains = rescued["would_abstain_new_threshold"].sum()
    print(f"\nHeld-out paraphrases (n={len(rescued)}): "
          f"{len(rescued) - still_abstains}/{len(rescued)} now correctly NOT abstaining "
          f"({still_abstains} still incorrectly abstain)", flush=True)

    oos_check = check_df[check_df["group"].str.contains("SHOULD_abstain")]
    correctly_abstains = oos_check["would_abstain_new_threshold"].sum()
    print(f"Fresh out-of-scope (n={len(oos_check)}): "
          f"{correctly_abstains}/{len(oos_check)} still correctly abstaining "
          f"({len(oos_check) - correctly_abstains} would now WRONGLY answer)", flush=True)

    print(f"\nWrote {OUT_HELDOUT_CHECK}")


if __name__ == "__main__":
    main()
