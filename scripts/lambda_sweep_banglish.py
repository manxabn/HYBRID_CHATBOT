"""
Retrieval-only lambda sweep on the Banglish test set: for each lambda in
{0.0, 0.1, ..., 1.0}, check whether the correct source row (the one the
reference_answer was actually drawn from) appears in the top-5 retrieved
chunks. This is cheap (no LLM generation needed) and directly answers
"does BM25 or vector deserve more weight for Banglish queries specifically",
extending the English-only lambda sweep already in paper.tex (Section 4.3)
with the language-stratified breakdown requested.

Correct-row matching is done by exact reference_answer string match against
each candidate chunk's text (since our corpus chunks are now one-row-per-
chunk after the 2026-07-27 rechunking fix, this is unambiguous).

Usage: python scripts/lambda_sweep_banglish.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever

QUERIES_PATH = ROOT / "data" / "test_queries_banglish.csv"
OUT_PATH = ROOT / "results" / "lambda_sweep_banglish.csv"

LAMBDAS = [round(x * 0.1, 1) for x in range(11)]


def main():
    df = pd.read_csv(QUERIES_PATH)
    retriever = HybridRetriever()

    rows = []
    for lam in LAMBDAS:
        n_hit = 0
        for _, r in df.iterrows():
            candidates = retriever.retrieve(r["query"], lam, top_n=5)
            hit = any(r["reference_answer"].strip() in c["text"] for c in candidates)
            n_hit += hit
        acc = n_hit / len(df)
        rows.append({"lambda": lam, "top5_accuracy": acc, "n": len(df)})
        print(f"lambda={lam}: top5_accuracy={acc:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    best = out.loc[out["top5_accuracy"].idxmax()]
    print(f"Best lambda: {best['lambda']} (top5_accuracy={best['top5_accuracy']:.3f})")


if __name__ == "__main__":
    main()
