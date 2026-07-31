"""
The BM25 k1/b tuning (Section on 2026-07-31 in CLAUDE.md.md) changed
retrieval ranking for ~55% of queries under bm25_only/full_hybrid, which
raised a real question: is the main ablation table's BLEU/ROUGE-L/
BERTScore/METEOR (results/ablation_metrics_summary.csv) now stale?

The full 800-generation pipeline was re-run post-tuning
(results/ablation_raw_outputs_post_bm25_tuning.csv, scored into
results/ablation_metrics_summary_post_bm25_tuning.csv). Eyeballing the
two summary tables shows tiny differences (<=0.0054 on any metric for
bm25_only/full_hybrid; vector_only/no_retrieval are byte-identical, as
expected since neither touches the BM25 index). This script confirms
that formally with a paired test per (config, metric) matched on
query_id, rather than trusting the eyeball read.

Usage: python scripts/verify_bm25_tuning_ablation_stability.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OLD_PATH = ROOT / "results" / "ablation_metrics_per_query.csv"
NEW_PATH = ROOT / "results" / "ablation_metrics_per_query_post_bm25_tuning.csv"
OUT_PATH = ROOT / "results" / "bm25_tuning_ablation_stability.csv"
METRICS = ["bleu", "rougeL", "bertscore", "meteor"]


def main():
    old = pd.read_csv(OLD_PATH)
    new = pd.read_csv(NEW_PATH)

    rows = []
    for config in sorted(old["config"].unique()):
        o = old[old["config"] == config].set_index("query_id")
        n = new[new["config"] == config].set_index("query_id")
        common = o.index.intersection(n.index)
        print(f"[{config}] matched {len(common)} query_ids (old n={len(o)}, new n={len(n)})")
        for metric in METRICS:
            diffs = (n.loc[common, metric] - o.loc[common, metric]).values
            mean_diff = float(np.mean(diffs))
            t_stat, p_val = stats.ttest_rel(n.loc[common, metric], o.loc[common, metric])
            rows.append({
                "config": config, "metric": metric, "n": len(common),
                "mean_diff_new_minus_old": round(mean_diff, 5),
                "t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 5),
                "significant_at_0.05": bool(p_val < 0.05),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
