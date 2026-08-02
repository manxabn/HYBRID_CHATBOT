"""
Re-runs bootstrap_faithfulness_significance.py's exact methodology
(imported, not reimplemented) against the roundP faithfulness
regeneration -- both faithfulness_sample_baselines_roundO.csv and
_novel_roundO.csv predated the prerequisite-chain corpus fix and Chroma
vector-index rebuild (Section subsec:ir-metrics); re-generated against
the identical 40/50-query sampling methodology (same seed=42) under the
current, corrected corpus/index.

Usage: python scripts/bootstrap_faithfulness_significance_roundP.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_faithfulness_significance import paired_bootstrap

OUT_PATH = ROOT / "results" / "faithfulness_bootstrap_significance_roundP.csv"


def main():
    base = pd.read_csv(ROOT / "results" / "faithfulness_sample_baselines_roundP_faithfulness.csv")
    novel = pd.read_csv(ROOT / "results" / "faithfulness_sample_novel_roundP_faithfulness.csv")
    base = base.dropna(subset=["faithfulness"])
    novel = novel.dropna(subset=["faithfulness"])

    baseline_configs = sorted(base["config"].unique())
    rows = []

    pivoted = base.pivot(index="query_id", columns="config", values="faithfulness")
    print(f"Baseline configs share {pivoted.dropna().shape[0]}/{len(pivoted)} query_ids across all four")
    for i, c1 in enumerate(baseline_configs):
        for c2 in baseline_configs[i + 1:]:
            paired = pivoted[[c1, c2]].dropna()
            mean_diff, lo, hi, p = paired_bootstrap(paired[c1], paired[c2])
            rows.append({
                "comparison": f"{c1}_vs_{c2}", "n": len(paired),
                "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "p_approx": round(p, 4), "significant": not (lo <= 0 <= hi),
            })

    novel_by_id = novel.set_index("query_id")["faithfulness"]
    for cfg in baseline_configs:
        base_by_id = base[base["config"] == cfg].set_index("query_id")["faithfulness"]
        shared_ids = novel_by_id.index.intersection(base_by_id.index)
        if len(shared_ids) < 5:
            print(f"Skipping adaptive_novel_vs_{cfg}: only {len(shared_ids)} shared query_ids, too few")
            continue
        mean_diff, lo, hi, p = paired_bootstrap(novel_by_id.loc[shared_ids], base_by_id.loc[shared_ids])
        rows.append({
            "comparison": f"adaptive_novel_vs_{cfg}", "n": len(shared_ids),
            "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "p_approx": round(p, 4), "significant": not (lo <= 0 <= hi),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
