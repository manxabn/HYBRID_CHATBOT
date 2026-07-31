"""
Re-scores avoids_false_confidence in results/conditioning_hint_v2_raw.csv
using the VALIDATED decomposed judge (kappa=0.7471, scripts/improve_avoids_
false_confidence_judge.py) instead of the original holistic judge
(kappa=0.114) that scripts/eval_conditioning_hint_v2.py used via its import
of judge_response_or_fail from eval_ambiguous_entity_notice_quality.py.

Found 2026-07-31 by manually inspecting conditioning_v2 responses scored
avoids_false_confidence=0: several were clear clarifying questions (e.g.
"Which one do you mean: C. Lecturer (Contractual) or Lecturer (Full
Time)?") wrongly marked as NOT avoiding false confidence -- the same
known near-chance-reliability failure mode already fixed for the main
660-row dataset earlier this session, never applied here since this
script/dataset didn't exist yet when that fix was made.

The other two criteria (asks_for_clarification kappa=0.984,
offers_disambiguator kappa=0.656) are NOT re-scored -- already reliable,
carried over as-is.

Usage: python scripts/apply_fixed_afc_judge_conditioning_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from improve_avoids_false_confidence_judge import extract_or_fail, EXTRACT_PROMPT_A

RAW_PATH = ROOT / "results" / "conditioning_hint_v2_raw.csv"
OUT_RAW_PATH = ROOT / "results" / "conditioning_hint_v2_raw_fixed_afc.csv"
OUT_SUMMARY_PATH = ROOT / "results" / "conditioning_hint_v2_summary_fixed_afc.csv"
OUT_SIGNIFICANCE_PATH = ROOT / "results" / "conditioning_hint_v2_significance_fixed_afc.csv"


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
    print(f"Re-scoring avoids_false_confidence for all {len(df)} responses with the "
          f"validated decomposed judge (kappa=0.7471)...")

    new_scores = []
    parse_failed = []
    for i, r in df.iterrows():
        result = extract_or_fail(EXTRACT_PROMPT_A, r["query"], r["answer"])
        if result is None:
            new_scores.append(None)
            parse_failed.append(True)
            print(f"  [{i+1}/{len(df)}] {r['query_id']}/{r['condition']}: PARSE FAILED, excluded", flush=True)
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
          f"EXCLUDED from the summary/significance below, not defaulted.")
    df = df[~df["parse_failed"]]

    changed = (df["avoids_false_confidence"] != df["avoids_false_confidence_OLD_holistic"]).mean()
    print(f"\nVerdict changed on {changed:.1%} of clean responses vs the old holistic judge")

    summary = df.groupby("condition")["avoids_false_confidence"].mean()
    summary.to_csv(OUT_SUMMARY_PATH)
    print(f"\n=== avoids_false_confidence, FIXED judge, n={len(df)} clean responses ===")
    print(summary)

    pivot = df.pivot(index="query_id", columns="condition", values="avoids_false_confidence")
    pivot = pivot.dropna(subset=["no_notice", "flat_notice", "conditioning", "conditioning_v2"])
    rows = []
    print(f"\n=== Paired bootstrap significance, FIXED avoids_false_confidence "
          f"(n={len(pivot)} query_ids complete across all 4 conditions) ===")
    for a, b in [("conditioning_v2", "no_notice"), ("conditioning_v2", "flat_notice"),
                 ("conditioning_v2", "conditioning")]:
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
