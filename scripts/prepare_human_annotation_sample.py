"""
Prepares the human hallucination-annotation sample for weakness #2, per the
disagreement-prioritized sampling strategy: rather than annotate a random
subsample (wasting labels on cases both automated methods already agree on),
prioritize the query_id's where the LLM-judge (RAGAS-style claim
decomposition) and the independent NLI-based check (SummaC-ZS, Section
nli-faithfulness) disagree most -- these are exactly the ambiguous,
informative cases where a second, human, rater's judgment adds the most
value, and where the two automated methods' r=0.089 correlation is telling
us the least.

This script does the MACHINE side only: computing per-row disagreement,
selecting the sample, and writing a clean annotation sheet with blank
columns for a human rater to fill in. It does NOT annotate anything itself
-- an LLM filling in its own hallucination judgments here would not
constitute the human annotation this project has repeatedly, explicitly
flagged as a gap it cannot close alone.

Sample size (40-60) and one-rater-plus-comparison design follow the
pattern documented in RAG evaluation literature for small-scale human
validation studies (e.g. per-file spot-check validation at similar scale
in published automatic-scorer validation sections).

Usage: python scripts/prepare_human_annotation_sample.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IN_PATH = ROOT / "results" / "nli_faithfulness_per_query.csv"
OUT_PATH = ROOT / "results" / "human_annotation_sample.csv"
SAMPLE_SIZE = 50


def main():
    df = pd.read_csv(IN_PATH)
    df = df.dropna(subset=["faithfulness", "nli_faithfulness"]).copy()
    df["disagreement"] = (df["faithfulness"] - df["nli_faithfulness"]).abs()

    # Prioritize disagreement, but keep at least a few low-disagreement rows
    # too (a pure disagreement-only sample can't report a meaningful overall
    # Cohen's kappa against either automated method, since it would exclude
    # every case they already agree on by construction) -- 40 highest-
    # disagreement rows + 10 agreement-anchor rows sampled from the rest.
    n_priority = min(40, len(df))
    priority = df.nlargest(n_priority, "disagreement")
    remaining = df.drop(priority.index)
    n_anchor = min(SAMPLE_SIZE - n_priority, len(remaining))
    anchor = remaining.sample(n=n_anchor, random_state=42) if n_anchor > 0 else remaining.iloc[0:0]

    sample = pd.concat([priority, anchor]).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle so order doesn't hint at priority-vs-anchor

    out = sample[[
        "query_id", "config", "query", "reference_answer", "retrieved_context",
        "generated_answer", "faithfulness", "nli_faithfulness", "disagreement",
    ]].rename(columns={"faithfulness": "llm_judge_score", "nli_faithfulness": "nli_score"})

    # Blank columns for the human rater to fill in -- deliberately empty,
    # not pre-filled with any automated guess.
    out["human_label"] = ""  # expected values: faithful / partial / hallucinated
    out["human_notes"] = ""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(f"  {n_priority} disagreement-prioritized rows (mean disagreement "
          f"{priority['disagreement'].mean():.3f}) + {n_anchor} agreement-anchor rows "
          f"(mean disagreement {anchor['disagreement'].mean():.3f} if anchor else 'n/a')")
    print("\nRubric to give the human rater (write this in your paper's annotation instructions):")
    print("  faithful      -- every claim in the answer is supported by the retrieved context")
    print("  partial       -- the answer is directionally correct but states an unsupported detail")
    print("  hallucinated  -- the answer states something the retrieved context does not support")
    print("\nNext step (not automatable): get one human rater (co-author/supervisor-approved "
          "helper) to fill in human_label for each row, independently of the llm_judge_score/"
          "nli_score columns shown here -- then compute Cohen's kappa between human_label and "
          "each automated method's binarized judgment.")


if __name__ == "__main__":
    main()
