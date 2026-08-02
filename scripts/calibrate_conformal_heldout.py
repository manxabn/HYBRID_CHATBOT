"""
Held-out validation of pipeline/conformal_abstention.py's calibration,
following exactly the same discipline this project already applied to the
open-ended abstention gate (Section "Limitations" in paper.tex): calibrating
a threshold and reporting its risk on the SAME data it was fit on measures
in-sample fit quality, not generalization -- scripts/run_conformal_
calibration.py does exactly that, and this script exists to check whether
its result survives a real train/test split before anything gets enabled
or claimed in the paper.

Also records the per-claim NLI score distribution split by label
(positive/verified-retrieval vs. negative/out-of-scope), since the
calibrated threshold alone doesn't explain WHY a result lands where it
does -- and here it doesn't land where a working confidence signal would
put it (see module docstring / paper.tex for the finding and mechanistic
explanation).

Usage: python scripts/calibrate_conformal_heldout.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.conformal_abstention import score_claims

LABELS_PATH = ROOT / "results" / "conformal_calibration_labels.csv"
SCORED_OUT = ROOT / "results" / "conformal_calibration_labels_scored.csv"
RESULT_OUT = ROOT / "results" / "conformal_calibration_heldout_result.json"
TARGET_RISK = 0.1
SPLIT_SEED = 42
TRAIN_FRAC = 0.7


def main():
    df = pd.read_csv(LABELS_PATH)
    if SCORED_OUT.exists():
        scored = pd.read_csv(SCORED_OUT)
        if len(scored) == len(df) and "nli_score" in scored.columns:
            df = scored
            print(f"Reusing cached NLI scores from {SCORED_OUT}", flush=True)
    if "nli_score" not in df.columns:
        print("Scoring claims against context with the NLI model (device=cpu)...", flush=True)
        scores = []
        for _, row in df.iterrows():
            s = score_claims([row["claim"]], row["context"], device="cpu")
            scores.append(s[0] if s else 0.0)
        df["nli_score"] = scores
        df.to_csv(SCORED_OUT, index=False)

    pos = df[df["is_correct"] == True]["nli_score"]
    neg = df[df["is_correct"] == False]["nli_score"]
    dist_summary = {
        "positive_mean": float(pos.mean()), "positive_median": float(pos.median()),
        "negative_mean": float(neg.mean()), "negative_median": float(neg.median()),
        "n_positive": int(len(pos)), "n_negative": int(len(neg)),
    }
    print(f"Positive (should-be-correct) claims: mean NLI={dist_summary['positive_mean']:.4f}, "
          f"median={dist_summary['positive_median']:.4f}, n={dist_summary['n_positive']}")
    print(f"Negative (should-be-incorrect) claims: mean NLI={dist_summary['negative_mean']:.4f}, "
          f"median={dist_summary['negative_median']:.4f}, n={dist_summary['n_negative']}")

    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(df))
    split = int(len(df) * TRAIN_FRAC)
    train_idx, test_idx = idx[:split], idx[split:]
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(f"\nTrain n={len(train)} ({int(train['is_correct'].sum())} positive), "
          f"test n={len(test)} ({int(test['is_correct'].sum())} positive)", flush=True)

    train_rows = list(zip(train["nli_score"], train["is_correct"]))
    candidate_thresholds = sorted({round(s, 4) for s, _ in train_rows})
    threshold, train_risk = 1.0, 0.0
    for lam in candidate_thresholds:
        retained = [(s, c) for s, c in train_rows if s >= lam]
        risk = 0.0 if not retained else sum(1 for s, c in retained if not c) / len(retained)
        if risk <= TARGET_RISK:
            threshold, train_risk = lam, risk
            break

    test_rows = list(zip(test["nli_score"], test["is_correct"]))
    retained_test = [(s, c) for s, c in test_rows if s >= threshold]
    retained_frac = len(retained_test) / len(test_rows)
    held_out_risk = (float("nan") if not retained_test
                      else sum(1 for s, c in retained_test if not c) / len(retained_test))

    result = {
        "target_risk": TARGET_RISK, "split_seed": SPLIT_SEED, "train_frac": TRAIN_FRAC,
        "n_train": len(train), "n_test": len(test),
        "train_calibrated_threshold": threshold, "train_achieved_risk": train_risk,
        "test_n_retained": len(retained_test), "test_n_total": len(test_rows),
        "test_retained_fraction": retained_frac, "test_held_out_risk": held_out_risk,
        "score_distribution": dist_summary,
        "conclusion": ("degenerate: held-out retained fraction is 0%, meaning this calibrated "
                       "threshold would empty-filter every open-ended answer in production" if retained_frac == 0
                       else "usable" if retained_frac > 0.1 else "marginal, very low retained fraction"),
    }
    RESULT_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nTrain-calibrated threshold: {threshold:.4f} (achieved risk on train: {train_risk:.4f})")
    print(f"Held-out test: retained {len(retained_test)}/{len(test_rows)} claims "
          f"({retained_frac:.1%}), risk among retained={held_out_risk}")
    print(f"Conclusion: {result['conclusion']}")
    print(f"\nWrote {SCORED_OUT} and {RESULT_OUT}")


if __name__ == "__main__":
    main()
