"""
Consumes results/conformal_calibration_labels.csv (built by scripts/build_
conformal_calibration_labels.py -- structural, non-self-judged labels: Out-
of-Scope-derived negatives, verified-retrieval-derived positives, see that
script's docstring for the full methodology and its honest caveats) and
runs pipeline/conformal_abstention.py's calibrate_threshold_from_labels()
to get a REAL calibrated threshold, replacing the current provisional
OPEN_ENDED_THRESHOLD=0.35 heuristic.

Reports the result honestly regardless of outcome: if the label set is too
small, too imbalanced (all-positive or all-negative), or the achieved risk
can't reach target_risk at any threshold, that is reported plainly, not
smoothed over -- this calibration is a genuine upgrade over "arbitrary
heuristic" but is explicitly NOT equivalent to human-validated calibration
on the harder disagreement-prioritized cases (results/human_annotation_
sample.csv, still unlabeled). Both facts are stated in the output, not just
in code comments, so this can't be silently oversold later.

Usage: python scripts/run_conformal_calibration.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.conformal_abstention import calibrate_threshold_from_labels

LABELS_PATH = ROOT / "results" / "conformal_calibration_labels.csv"
OUT_PATH = ROOT / "results" / "conformal_calibration_result.json"
TARGET_RISK = 0.1


def main():
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} not found -- run scripts/build_conformal_calibration_labels.py first.")

    df = pd.read_csv(LABELS_PATH)
    n_pos = int(df["is_correct"].sum())
    n_neg = int((~df["is_correct"]).sum())
    print(f"Loaded {len(df)} labeled claims ({n_pos} positive / verified_retrieval, "
          f"{n_neg} negative / out_of_scope)")

    if len(df) < 20:
        print(f"WARNING: n={len(df)} is very small for calibration -- treat any resulting "
              f"threshold as a weak, indicative starting point, not a reliable calibrated value.")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(
            f"Cannot calibrate: label set has {n_pos} positive and {n_neg} negative examples -- "
            f"calibrate_threshold_from_labels needs both classes represented. This is a real, "
            f"honestly-reported dead end for this specific labeling approach, not a bug to route around "
            f"by fabricating the missing class."
        )

    labeled_rows = df[["claim", "context", "is_correct"]].to_dict("records")
    print("Scoring claims against context with the NLI model (device=cpu) -- this takes a few minutes...")
    result = calibrate_threshold_from_labels(labeled_rows, target_risk=TARGET_RISK, device="cpu")

    result["n_positive"] = n_pos
    result["n_negative"] = n_neg
    result["target_risk"] = TARGET_RISK
    result["label_source"] = "structural (Out-of-Scope negatives + verified-retrieval positives), NOT human-annotated"
    result["not_covered"] = ("results/human_annotation_sample.csv's 50 disagreement-prioritized cases remain "
                              "unlabeled and are NOT represented in this calibration set -- those were "
                              "specifically selected as genuinely hard/ambiguous cases this structural "
                              "approach cannot resolve; this threshold has NOT been validated against them.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibrated threshold: {result['threshold']:.4f}")
    print(f"Achieved risk on calibration set: {result['achieved_risk']:.4f} (target: {TARGET_RISK})")
    print(f"n={result['n']} ({n_pos} positive / {n_neg} negative)")
    print(f"\nWrote {OUT_PATH}")
    print("\nHonest scope note: this threshold is calibrated against structurally-derived labels, "
          "not human judgment. It is a real, disclosed improvement over the previous arbitrary 0.35 "
          "heuristic, but is not the human-validated guarantee a full Mohri-Hashimoto-style claim would need.")


if __name__ == "__main__":
    main()
