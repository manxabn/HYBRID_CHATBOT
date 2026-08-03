"""
Root-cause investigation into WHY the NLI-vs-LLM-judge Pearson r stays
near zero (and slightly negative) across all three completed NLI model
swaps this session, using data already on disk -- no new generation or
NLI scoring, just analysis of the existing per-query CSVs.

Checks several concrete, testable hypotheses rather than speculating:
  1. Restricted range: if either instrument's scores cluster tightly
     (low variance), correlation is mechanically suppressed regardless
     of whether the instruments actually agree on relative ordering.
  2. Per-config breakdown: is the (near-)zero correlation uniform across
     configs, or driven by one particular retrieval configuration?
  3. Ceiling effects: does the LLM judge score almost everything high
     (a known RAGAS-style judge tendency), leaving little room for any
     external signal to correlate with?
  4. Concrete disagreement cases: the specific rows where the two
     instruments disagree most, inspected directly (query, context,
     answer, judge's own stated reason, NLI score) -- to characterize
     the disagreement qualitatively, not just statistically.

Usage: python scripts/diagnose_nli_correlation_roundP.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FILES = {
    "MiniLM": (ROOT / "results" / "nli_faithfulness_per_query_roundP.csv", "nli_faithfulness"),
    "DeBERTa-base": (ROOT / "results" / "nli_faithfulness_per_query_roundP_deberta.csv", "nli_faithfulness"),
    "DeBERTa-large-FEVER": (ROOT / "results" / "nli_faithfulness_per_query_roundP_fever.csv", "nli_faithfulness"),
}


def main():
    for model_name, (path, col) in FILES.items():
        if not path.exists():
            print(f"SKIP {model_name}: {path} not found")
            continue
        df = pd.read_csv(path)
        both = df.dropna(subset=["faithfulness", col])
        print(f"\n{'='*70}\n{model_name} (n={len(both)})\n{'='*70}")

        print(f"LLM judge (faithfulness):  mean={both['faithfulness'].mean():.4f}  "
              f"std={both['faithfulness'].std():.4f}  "
              f"range=[{both['faithfulness'].min():.3f}, {both['faithfulness'].max():.3f}]  "
              f"pct>=0.9={100*(both['faithfulness']>=0.9).mean():.1f}%")
        print(f"NLI check ({col}):        mean={both[col].mean():.4f}  "
              f"std={both[col].std():.4f}  "
              f"range=[{both[col].min():.3f}, {both[col].max():.3f}]  "
              f"pct>=0.9={100*(both[col]>=0.9).mean():.1f}%")

        r = both["faithfulness"].corr(both[col])
        print(f"Overall Pearson r: {r:.4f}")

        if "config" in both.columns:
            print("Per-config breakdown:")
            for cfg, grp in both.groupby("config"):
                if len(grp) >= 5:
                    rc = grp["faithfulness"].corr(grp[col])
                    print(f"  {cfg:20s} n={len(grp):3d}  r={rc:+.4f}  "
                          f"judge_mean={grp['faithfulness'].mean():.3f}  nli_mean={grp[col].mean():.3f}")

        # Spearman (rank) correlation -- if Pearson is suppressed by a
        # nonlinear relationship or outliers but the two still agree on
        # RELATIVE ordering, Spearman would show that even if Pearson doesn't.
        rs = both["faithfulness"].corr(both[col], method="spearman")
        print(f"Spearman rank correlation: {rs:.4f} (compare to Pearson {r:.4f} above -- "
              f"if Spearman is meaningfully higher, the relationship is monotonic but nonlinear, "
              f"not simply absent)")

    # Deep dive on the best (FEVER) model's biggest disagreements
    path, col = FILES["DeBERTa-large-FEVER"]
    if path.exists():
        df = pd.read_csv(path)
        both = df.dropna(subset=["faithfulness", col]).copy()
        both["disagreement"] = (both["faithfulness"] - both[col]).abs()
        print(f"\n{'='*70}\nTop 5 disagreeing rows (DeBERTa-large-FEVER, judge vs NLI)\n{'='*70}")
        top = both.sort_values("disagreement", ascending=False).head(5)
        for _, row in top.iterrows():
            print(f"\n--- query_id={row.get('query_id')} config={row.get('config')} "
                  f"judge={row['faithfulness']:.2f} nli={row[col]:.2f} ---")
            print(f"Query: {str(row.get('query'))[:150]}")
            print(f"Answer: {str(row.get('generated_answer'))[:200]}")
            print(f"Judge reason: {str(row.get('faithfulness_reason'))[:250]}")
            ctx = str(row.get('retrieved_context'))
            print(f"Context (first 200 chars): {ctx[:200]}")


if __name__ == "__main__":
    main()
