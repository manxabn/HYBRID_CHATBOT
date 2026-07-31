"""
Post-hoc statistical power for this project's small-n paired ablations
(graph augmentation, n=12; cross-lingual stress test, n=9/27), flagged as a
real weakness in this session's own codebase audit: a significant-looking
p-value from a small n can still reflect a study with low power to have
detected the effect in the first place, which inflates confidence in the
observed effect size (a well-documented statistics concern, independent of
any single paper -- computed here directly via scipy rather than citing an
unverified secondary source for the general concept).

Method: standard post-hoc power for a paired t-test, computed from the
paired differences actually observed (not assumed): Cohen's d_z = mean_diff
/ std_diff, noncentrality parameter ncp = d_z * sqrt(n), then power =
P(reject H0) under the noncentral t-distribution with df=n-1 and that ncp,
at a two-sided alpha=0.05 critical value. Uses scipy.stats.nct directly
(statsmodels is not installed in this project's venv and this avoids
adding a new dependency for one calculation).

Usage: python scripts/power_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALPHA = 0.05


def paired_power(diffs: np.ndarray, alpha: float = ALPHA) -> dict:
    """diffs: array of (condition_a - condition_b) per matched pair."""
    n = len(diffs)
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    if std_diff == 0:
        return {"n": n, "cohens_dz": float("nan"), "power": float("nan"),
                "note": "zero variance in paired differences, power undefined"}
    dz = mean_diff / std_diff
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = dz * np.sqrt(n)
    # Two-sided power: P(T > t_crit) + P(T < -t_crit) under noncentral t(df, ncp)
    power = (1 - stats.nct.cdf(t_crit, df, ncp)) + stats.nct.cdf(-t_crit, df, ncp)
    return {"n": n, "mean_diff": mean_diff, "cohens_dz": dz, "power": power}


def required_n_for_power(dz: float, target_power: float = 0.8, alpha: float = ALPHA, max_n: int = 2000) -> int:
    """Smallest n (paired samples) such that a paired t-test would reach
    target_power at the GIVEN observed effect size dz -- i.e. "how many
    more of these would we need to collect to trust this effect," not a
    guess. Grid search rather than a closed form since nct's CDF has no
    simple inverse in n."""
    for n in range(2, max_n):
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ncp = dz * np.sqrt(n)
        power = (1 - stats.nct.cdf(t_crit, df, ncp)) + stats.nct.cdf(-t_crit, df, ncp)
        if power >= target_power:
            return n
    return -1  # not reached within max_n


def report(label: str, diffs: np.ndarray):
    r = paired_power(diffs)
    print(f"\n{label}")
    print(f"  n={r['n']}, mean_diff={r.get('mean_diff', float('nan')):.4f}, "
          f"Cohen's d_z={r['cohens_dz']:.3f}")
    print(f"  Achieved power (alpha=0.05, two-sided): {r['power']:.3f}"
          + ("  <-- below the conventional 0.8 threshold" if r["power"] < 0.8 else ""))
    if r["power"] < 0.8 and not np.isnan(r["cohens_dz"]) and r["cohens_dz"] != 0:
        needed = required_n_for_power(abs(r["cohens_dz"]))
        print(f"  n needed for 80% power AT THIS OBSERVED EFFECT SIZE: {needed} "
              f"(currently have {r['n']})")


def main():
    # Graph augmentation ablation (paper.tex subsec:graph, n=12): BLEU was
    # the metric with the strongest reported effect (p=0.0995 pre-fix
    # comparison notwithstanding -- this recomputes power on the ACTUAL
    # post-fix per-query BLEU values in the raw metrics file).
    graph_path = ROOT / "results" / "graph_ablation_metrics.csv"
    if graph_path.exists():
        df = pd.read_csv(graph_path)
        on = df[df["config"] == "graph_on"].set_index("query_id")["bleu"]
        off = df[df["config"] == "graph_off"].set_index("query_id")["bleu"]
        common = on.index.intersection(off.index)
        report("Graph augmentation ablation (BLEU, graph_on - graph_off)",
               (on.loc[common] - off.loc[common]).values)
    else:
        print(f"Skipping graph augmentation: {graph_path} not found")

    # Cross-lingual stress test (paper.tex subsec:crosslingual-stress):
    # binary top-5-hit per query, original n=9. McNemar-style discordant-
    # pairs test was used in the paper for the binary outcome; a paired
    # t-test on the 0/1 hit indicators is the continuous-approximation
    # equivalent power calculation for the same paired design.
    for label, fname in [("Cross-lingual stress test, original (n=9)", "crosslingual_stress_eval.csv"),
                          ("Cross-lingual stress test, expanded (n=27)", "crosslingual_stress_eval_expanded.csv")]:
        stress_path = ROOT / "results" / fname
        if not stress_path.exists():
            print(f"\nSkipping {label}: {stress_path} not found")
            continue
        df = pd.read_csv(stress_path)
        diffs = (df["translated_top5_hit"].astype(int) - df["plain_top5_hit"].astype(int)).values
        report(f"{label}: translated_hit - plain_hit", diffs)


if __name__ == "__main__":
    main()
