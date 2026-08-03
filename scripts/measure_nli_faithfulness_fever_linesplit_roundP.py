"""
A real methodological fix attempt, following directly from the root
-cause investigation in diagnose_nli_correlation_roundP.py: rows Q081
and Q184 showed the LLM judge correctly scoring an answer as fully
faithful (a fact copied verbatim from a structured multi-field context
block, e.g. an email address from a FacultyList record, or list
-membership in "10 types of scholarship") while the NLI check scored
both near zero.

Root cause: this corpus's structured-table-derived chunks (Coordinator/
FacultyList/CourseDetails/FacultyAvailability) are formatted as
"Field: value\\nField: value\\n..." -- nltk.sent_tokenize (designed for
PROSE) does not reliably split these on the field boundaries, so a
single "sentence" can contain several unrelated fields glued together,
diluting the entailment signal for whichever specific field actually
supports the claim.

Fix: split context into candidate premises by BOTH newline AND sentence
boundaries (nltk.sent_tokenize applied to each line), so a structured
record's individual fields become separate, cleaner premise candidates
alongside prose sentences -- the same SummaC-ZS max-aggregation logic
otherwise unchanged, just given better-segmented input for HALF the
corpus's chunk types (the structured tables) without changing anything
for the other half (already-prose QA chunks, where sentence splitting
was already appropriate).

Reuses the winning FEVER-augmented model (best of the three tested:
r=-0.018) and GPU (confirmed free, same verification as the other GPU
runs tonight) on the identical 169-row sample, changing ONLY the context
-side segmentation, for a direct, controlled before/after comparison.

Usage: python scripts/measure_nli_faithfulness_fever_linesplit_roundP.py
"""

import sys
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
LABELS = ["entailment", "neutral", "contradiction"]
DEVICE = "cuda"
BATCH_SIZE = 16

INPUT_FILES = {
    "baselines": ROOT / "results" / "faithfulness_sample_baselines_roundP_faithfulness.csv",
    "novel": ROOT / "results" / "faithfulness_sample_novel_roundP_faithfulness.csv",
}
OUT_PER_QUERY = ROOT / "results" / "nli_faithfulness_per_query_roundP_fever_linesplit.csv"
OUT_SUMMARY = ROOT / "results" / "nli_faithfulness_summary_roundP_fever_linesplit.csv"
OUT_AGREEMENT = ROOT / "results" / "nli_vs_llmjudge_agreement_roundP_fever_linesplit.csv"


def context_premises(context: str) -> list[str]:
    """Line-then-sentence split -- see module docstring for why."""
    premises = []
    for line in context.split("\n"):
        line = line.strip()
        if not line:
            continue
        premises.extend(nltk.sent_tokenize(line))
    return premises or [""]


def main():
    print(f"Loading {NLI_MODEL} on {DEVICE}...")
    model = CrossEncoder(NLI_MODEL, device=DEVICE)

    frames = []
    for label, path in INPUT_FILES.items():
        if not path.exists():
            print(f"Skipping {label}: {path} not found")
            continue
        df = pd.read_csv(path)
        df["sample_source"] = label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["retrieved_context"] = df.get("retrieved_context", "").fillna("")
    df["generated_answer"] = df["generated_answer"].fillna("")
    scoreable = df[
        (df["retrieved_context"].str.strip() != "") &
        (df["generated_answer"].str.strip() != "") &
        (df.get("abstained", False).astype(str).str.lower() != "true")
    ].copy()
    print(f"Scoring {len(scoreable)}/{len(df)} rows (rest skipped: no context/answer/abstained)")

    row_pair_ranges = []
    all_pairs = []
    for idx, row in scoreable.iterrows():
        context_sents = context_premises(str(row["retrieved_context"]))
        answer_sents = nltk.sent_tokenize(str(row["generated_answer"])) or [""]
        start = len(all_pairs)
        for a_sent in answer_sents:
            for c_sent in context_sents:
                all_pairs.append((c_sent, a_sent))
        row_pair_ranges.append((idx, len(answer_sents), len(context_sents), start))

    print(f"Scoring {len(all_pairs)} (context-premise, answer-sentence) pairs "
          f"(line+sentence split, vs. sentence-only in the original run)...")
    with torch.no_grad():
        logits = model.predict(all_pairs, convert_to_numpy=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    p_entailment = probs[:, LABELS.index("entailment")]

    row_scores = {}
    for idx, n_answer, n_context, start in row_pair_ranges:
        per_answer_sent_max = []
        for a in range(n_answer):
            seg = p_entailment[start + a * n_context: start + (a + 1) * n_context]
            per_answer_sent_max.append(seg.max() if len(seg) else 0.0)
        row_scores[idx] = float(sum(per_answer_sent_max) / len(per_answer_sent_max)) if per_answer_sent_max else 0.0
    scoreable["nli_faithfulness_linesplit"] = scoreable.index.map(row_scores)

    OUT_PER_QUERY.parent.mkdir(parents=True, exist_ok=True)
    scoreable.to_csv(OUT_PER_QUERY, index=False)
    print(f"Wrote per-query to {OUT_PER_QUERY}")

    group_col = "config" if "config" in scoreable.columns else "sample_source"
    summary = scoreable.groupby(group_col)["nli_faithfulness_linesplit"].agg(["mean", "count"])
    summary.to_csv(OUT_SUMMARY)
    print(summary)

    if "faithfulness" in scoreable.columns:
        both = scoreable.dropna(subset=["faithfulness", "nli_faithfulness_linesplit"])
        corr = both["faithfulness"].corr(both["nli_faithfulness_linesplit"])
        agree = ((both["faithfulness"] >= 0.5) == (both["nli_faithfulness_linesplit"] >= 0.5)).mean()
        pd.DataFrame([{"n_both_scored": len(both), "pearson_r_llmjudge_vs_nli_linesplit": corr,
                        "binary_agreement_at_0.5": agree}]).to_csv(OUT_AGREEMENT, index=False)
        print(f"\nLLM-judge vs NLI (line-split context) agreement (n={len(both)}):")
        print(f"  Pearson r: {corr:.4f}  (compare to sentence-only split: r=-0.0177)")
        print(f"  Binary agreement: {agree:.3f}  (compare to sentence-only split: 0.905)")
        print(f"Wrote {OUT_AGREEMENT}")


if __name__ == "__main__":
    main()
