"""
McNemar's exact test for adaptive vs. an external late-interaction baseline
on the binary IR metrics (recall@1/3/5), companion to the paired bootstrap
in scripts/eval_colbert_baseline.py -- this project's established
convention for binary paired metrics is to require BOTH a paired bootstrap
and McNemar's test to agree before calling a result significant (see
scripts/mcnemar_full_hybrid_vs_bm25.py for the same convention applied to
an earlier comparison). Reused verbatim, not reimplemented.

Parameterized over --label/--raw-csv so the same test runs against more
than one baseline (colbert_external, gte_moderncolbert, ...).

Usage:
  python scripts/mcnemar_adaptive_vs_colbert.py --label colbert_external
  python scripts/mcnemar_adaptive_vs_colbert.py --label gte_moderncolbert
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IR_METRICS_PATH = ROOT / "results" / "ir_metrics.csv"
METRICS = ["recall@1", "recall@3", "recall@5"]


def mcnemar(a, b):
    a_wins = int(((a == 1) & (b == 0)).sum())
    b_wins = int(((a == 0) & (b == 1)).sum())
    n = a_wins + b_wins
    if n == 0:
        return a_wins, b_wins, n, 1.0
    p = binomtest(min(a_wins, b_wins), n, 0.5, alternative="two-sided").pvalue
    return a_wins, b_wins, n, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="colbert_external")
    args = parser.parse_args()

    raw_path = ROOT / "results" / f"{args.label}_baseline_raw.csv"
    out_path = ROOT / "results" / f"mcnemar_adaptive_vs_{args.label}.csv"

    ir = pd.read_csv(IR_METRICS_PATH)
    adaptive = ir[ir.config == "adaptive"].set_index("query_id")
    other = pd.read_csv(raw_path).set_index("query_id")
    common = adaptive.index.intersection(other.index)
    print(f"n={len(common)} paired queries vs. {args.label}")

    rows = []
    for subset_name, mask_fn in [
        ("all", lambda idx: idx),
        ("entity_heavy", lambda idx: [i for i in idx if adaptive.loc[i, "is_entity_heavy"]]),
        ("open_ended", lambda idx: [i for i in idx if not adaptive.loc[i, "is_entity_heavy"]]),
    ]:
        idx = mask_fn(common)
        for metric in METRICS:
            a = adaptive.loc[idx, metric].values
            b = other.loc[idx, metric].values
            a_wins, b_wins, n_disc, p = mcnemar(a, b)
            rows.append({
                "subset": subset_name, "metric": metric, "n": len(idx),
                "adaptive_only_correct": a_wins, f"{args.label}_only_correct": b_wins,
                "n_discordant": n_disc, "mcnemar_p": round(p, 4),
                "significant_at_0.05": p < 0.05,
            })
            print(f"[{subset_name}] {metric}: adaptive-only-correct={a_wins}, "
                  f"{args.label}-only-correct={b_wins}, n_discordant={n_disc}, p={p:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
