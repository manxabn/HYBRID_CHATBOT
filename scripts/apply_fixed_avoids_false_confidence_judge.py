"""
Applies the VALIDATED decomposed avoids_false_confidence judge (scripts/
improve_avoids_false_confidence_judge.py -- confirmed kappa 0.114 -> 0.7548
on a 150-item paraphrase-invariance check) to the FULL 660-response
ambiguous-entity dataset (results/ambiguous_notice_quality_expanded_raw.csv),
replacing the unreliable holistic-judgment score with the reliable
extraction-based one everywhere this project reports avoids_false_
confidence.

Uses prompt A only (not the A/B reliability-testing pair) since that
validation is already done -- this is the actual scoring pass, not another
reliability check. Recomputes the full summary + paired-bootstrap
significance for avoids_false_confidence specifically (the other two
criteria, asks_for_clarification and offers_disambiguator, were already
reliable and are UNCHANGED, carried over as-is from the original raw file).

Usage: python scripts/apply_fixed_avoids_false_confidence_judge.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from improve_avoids_false_confidence_judge import extract_or_fail, EXTRACT_PROMPT_A

RAW_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_raw.csv"
OUT_RAW_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_raw_fixed_afc.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_summary_fixed_afc.csv"
OUT_SIGNIFICANCE_PATH = ROOT / "results" / "ambiguous_notice_quality_expanded_significance_fixed_afc.csv"


def bootstrap_ci_diff(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), lo, hi, p


def main():
    df = pd.read_csv(RAW_PATH)
    # 2026-07-31: the kappa=0.7548 validation this re-score was built on is
    # of UNCERTAIN trustworthiness (raw judge text wasn't logged for that
    # run, so 28/150 zero/zero rows can't be retroactively distinguished
    # from silent parse-failure defaults) -- do not cite that number as
    # solid, and re-validate with the fixed parser (n=150, single job)
    # before relying on this re-score's results either.
    print(f"Re-scoring avoids_false_confidence for all {len(df)} responses with the "
          f"decomposed judge (parse-failure-detecting; validation kappa needs re-confirming)...")

    new_scores = []
    parse_failed = []
    for i, r in df.iterrows():
        result = extract_or_fail(EXTRACT_PROMPT_A, r["query"], r["answer"])
        if result is None:
            new_scores.append(None)
            parse_failed.append(True)
            print(f"  [{i+1}/{len(df)}] {r['query_id']}: PARSE FAILED, excluded", flush=True)
        else:
            new_scores.append(result["avoids_false_confidence"])
            parse_failed.append(False)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} rescored", flush=True)

    df["avoids_false_confidence_OLD_holistic"] = df["avoids_false_confidence"]
    df["avoids_false_confidence"] = new_scores
    df["parse_failed"] = parse_failed
    df.to_csv(OUT_RAW_PATH, index=False)

    n_parse_failed = sum(parse_failed)
    print(f"\n{n_parse_failed}/{len(df)} rows FAILED TO PARSE even after retries -- "
          f"EXCLUDED from the summary/significance below, not defaulted to any score.")
    df = df[~df["parse_failed"]]

    criteria = ["avoids_false_confidence", "asks_for_clarification", "offers_disambiguator"]
    summary = df.groupby("condition")[criteria].mean()
    summary.to_csv(OUT_SUMMARY_PATH)
    print(f"\n=== Summary with FIXED avoids_false_confidence (mean, 0-1), n={len(df)} clean responses ===")
    print(summary)

    print(f"\n=== How much changed vs the old holistic judge ===")
    changed = (df["avoids_false_confidence"] != df["avoids_false_confidence_OLD_holistic"]).mean()
    print(f"Verdict changed on {changed:.1%} of the {len(df)} clean (non-parse-failed) responses")

    pivot = df.pivot(index="query_id", columns="condition", values="avoids_false_confidence")
    # A parse failure on just one of the 3 conditions for a given query_id
    # leaves NaN in that cell after pivoting -- drop any query_id that isn't
    # complete across all 3 conditions so the paired bootstrap below compares
    # genuinely paired arrays, not silently-misaligned ones.
    pivot = pivot.dropna(subset=["no_notice", "flat_notice", "conditioning"])
    rows = []
    print(f"\n=== Paired bootstrap significance, FIXED avoids_false_confidence "
          f"(n={len(pivot)} query_ids complete across all 3 conditions) ===")
    for a, b in [("flat_notice", "no_notice"), ("conditioning", "no_notice"), ("conditioning", "flat_notice")]:
        mean_diff, lo, hi, pval = bootstrap_ci_diff(pivot[a].values, pivot[b].values)
        sig = not (lo <= 0 <= hi)
        rows.append({"comparison": f"{a}_vs_{b}", "diff": round(mean_diff, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(pval, 4), "significant": sig})
        print(f"avoids_false_confidence: {a} vs {b}: diff={mean_diff:.3f} CI=[{lo:.3f},{hi:.3f}] "
              f"p={pval:.4f} significant={sig}")

    pd.DataFrame(rows).to_csv(OUT_SIGNIFICANCE_PATH, index=False)
    print(f"\nWrote {OUT_RAW_PATH}, {OUT_SUMMARY_PATH}, {OUT_SIGNIFICANCE_PATH}")


if __name__ == "__main__":
    main()
