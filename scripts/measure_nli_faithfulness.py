"""
Independent, architecturally-separate faithfulness measurement, motivated
directly by the HALT-RAG line of work (arXiv:2509.07475, 2025) found in
this session's literature survey: the existing faithfulness score
(scripts/compute_faithfulness.py) is a RAGAS-style LLM-as-judge, but the
judge is the SAME local model that does generation -- a disclosed
limitation (self-judging, no second/independent rater). An off-the-shelf
NLI (natural language inference) model never trained on this corpus, and
architecturally unrelated to the generation LLM, gives a genuinely
independent faithfulness signal: does the retrieved context ENTAIL the
generated answer? This is a simplified single-model version of HALT-RAG's
ensemble (they combine multiple NLI models + lexical signals); disclosed
as such, not presented as a full reproduction.

Runs entirely on the existing faithfulness sample CSVs (already have
retrieved_context + generated_answer from a prior LLM-judge run) -- no new
LLM generation calls, no Ollama involved, so this cannot interfere with or
slow down any concurrently-running Ollama-bound job.

Method: sentence-level decomposition + max-aggregation over NLI entailment
(the SummaC-ZS approach, Laban, Schnabel, Bennett & Hearst 2022, TACL --
"SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in
Summarization" -- adopted here rather than a naive single-shot
whole-document NLI call). A FIRST version of this script did premise=full
retrieved_context, hypothesis=full generated_answer in one NLI call per row
and got faithfulness scores of ~0.2-0.25 with near-zero correlation to the
LLM-judge score (Pearson r=0.034, 170 rows) -- a red flag, not a real
finding: standard NLI cross-encoders are trained on SHORT, single-sentence
premise/hypothesis pairs (SNLI/MultiNLI), and are known in the literature
to perform poorly at "does this entire multi-paragraph context entail this
entire multi-claim answer" in one shot. The fix, per SummaC-ZS: split the
context into sentences and the answer into sentences, and for each answer
sentence take the MAX entailment probability across all context sentences
(does *some* part of the context support *this specific claim*), then
average across answer sentences. This is much closer to what the RAGAS-
style LLM judge is actually doing (per-claim support checking) and is the
standard, citable way to apply sentence-pair-trained NLI models to
document-level faithfulness.

Usage: python scripts/measure_nli_faithfulness.py
"""

import re
import sys
from pathlib import Path

import nltk
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

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
# Switched from nli-deberta-v3-base (2026-07-28): system RAM was too tight
# (a concurrently-running Ollama-bound ablation job was already using most
# of it, and the instruction for this session was not to interfere with
# it) to reliably load the larger model. This MiniLM-based cross-encoder is
# a real, standard pretrained NLI model (much smaller footprint), trained
# on the same SNLI+MultiNLI data -- a reasonable substitution for this
# purpose, not a different task.
# Label order for this specific model's output head (contradiction, entailment, neutral) --
# confirmed against the model card, not assumed.
LABELS = ["contradiction", "entailment", "neutral"]

INPUT_FILES = {
    "baselines": ROOT / "results" / "faithfulness_sample_baselines_faithfulness.csv",
    "novel": ROOT / "results" / "faithfulness_sample_novel_faithfulness.csv",
}
OUT_PER_QUERY = ROOT / "results" / "nli_faithfulness_per_query.csv"
OUT_SUMMARY = ROOT / "results" / "nli_faithfulness_summary.csv"
OUT_AGREEMENT = ROOT / "results" / "nli_vs_llmjudge_agreement.csv"


def main():
    # Deliberately CPU: a background Ollama-bound job may be using the GPU
    # concurrently, and the instruction for this session was not to do
    # anything that could interfere with it. This model is small enough
    # (~440MB) that CPU inference on ~210 rows finishes in well under a
    # minute -- not worth the risk of GPU contention for the time saved.
    device = "cpu"
    print(f"Loading {NLI_MODEL} on {device}...")
    model = CrossEncoder(NLI_MODEL, device=device)

    frames = []
    for label, path in INPUT_FILES.items():
        if not path.exists():
            print(f"Skipping {label}: {path} not found")
            continue
        df = pd.read_csv(path)
        df["sample_source"] = label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Same skip logic as compute_faithfulness.py: nothing to check for
    # abstained rows or rows with no context/answer.
    df["retrieved_context"] = df.get("retrieved_context", "").fillna("")
    df["generated_answer"] = df["generated_answer"].fillna("")
    scoreable = df[
        (df["retrieved_context"].str.strip() != "") &
        (df["generated_answer"].str.strip() != "") &
        (df.get("abstained", False).astype(str).str.lower() != "true")
    ].copy()
    print(f"Scoring {len(scoreable)}/{len(df)} rows (rest skipped: no context/answer/abstained)")

    # Build ALL (context_sentence, answer_sentence) pairs across all rows in
    # one batch, so the model runs once rather than once per row -- context
    # sentences are typically short (a chunk's own sentences), well within
    # any per-pair length limit, so no truncation is needed at this level.
    row_pair_ranges = []  # (row_idx, n_answer_sents, n_context_sents)
    all_pairs = []
    for idx, row in scoreable.iterrows():
        context_sents = nltk.sent_tokenize(str(row["retrieved_context"])) or [""]
        answer_sents = nltk.sent_tokenize(str(row["generated_answer"])) or [""]
        start = len(all_pairs)
        for a_sent in answer_sents:
            for c_sent in context_sents:
                all_pairs.append((c_sent, a_sent))
        row_pair_ranges.append((idx, len(answer_sents), len(context_sents), start))

    print(f"Scoring {len(all_pairs)} (context-sentence, answer-sentence) pairs...")
    with torch.no_grad():
        logits = model.predict(all_pairs, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    p_entailment = probs[:, LABELS.index("entailment")]

    row_scores = {}
    for idx, n_answer, n_context, start in row_pair_ranges:
        # For each answer sentence, the max entailment prob over all context
        # sentences (SummaC-ZS "does *some* context sentence support *this*
        # claim"); the row's score is the mean of that over answer sentences.
        per_answer_sent_max = []
        for a in range(n_answer):
            seg = p_entailment[start + a * n_context: start + (a + 1) * n_context]
            per_answer_sent_max.append(seg.max() if len(seg) else 0.0)
        row_scores[idx] = float(sum(per_answer_sent_max) / len(per_answer_sent_max)) if per_answer_sent_max else 0.0

    scoreable["nli_faithfulness"] = scoreable.index.map(row_scores)

    OUT_PER_QUERY.parent.mkdir(parents=True, exist_ok=True)
    scoreable.to_csv(OUT_PER_QUERY, index=False)
    print(f"Wrote per-query NLI faithfulness to {OUT_PER_QUERY}")

    group_col = "config" if "config" in scoreable.columns else "sample_source"
    summary = scoreable.groupby(group_col)["nli_faithfulness"].agg(["mean", "count"])
    summary.to_csv(OUT_SUMMARY)
    print(f"Wrote summary to {OUT_SUMMARY}")
    print(summary)

    if "faithfulness" in scoreable.columns:
        both = scoreable.dropna(subset=["faithfulness", "nli_faithfulness"])
        corr = both["faithfulness"].corr(both["nli_faithfulness"])
        # Binary agreement at a 0.5 threshold on each scale -- a simple,
        # interpretable inter-method agreement statistic (not Cohen's kappa,
        # since these are continuous scores from two different scales/
        # methods, not two raters applying the same categorical rubric).
        agree = ((both["faithfulness"] >= 0.5) == (both["nli_faithfulness"] >= 0.5)).mean()
        agreement_row = pd.DataFrame([{
            "n_both_scored": len(both),
            "pearson_r_llmjudge_vs_nli": corr,
            "binary_agreement_at_0.5": agree,
        }])
        agreement_row.to_csv(OUT_AGREEMENT, index=False)
        print(f"\nLLM-judge vs NLI-judge agreement (n={len(both)}):")
        print(f"  Pearson r: {corr:.3f}")
        print(f"  Binary agreement (both >=0.5 or both <0.5): {agree:.3f}")
        print(f"Wrote {OUT_AGREEMENT}")


if __name__ == "__main__":
    main()
