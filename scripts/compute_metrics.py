"""
Score results/ablation_raw_outputs.csv against reference_answer with
BLEU, ROUGE-L, BERTScore, and METEOR, and write:
  - results/ablation_metrics_per_query.csv (row-level)
  - results/ablation_metrics_summary.csv   (mean per config -- Table 1 source)

To score a different raw-outputs file (e.g. the RRF novelty run from
`python scripts/run_ablation.py --fusion rrf`), pass --raw/--per-query-out/
--summary-out; defaults are unchanged so the existing linear-fusion results
are untouched.
"""

import argparse
import sys
from pathlib import Path

import nltk
import pandas as pd
import torch
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = ROOT / "results" / "ablation_raw_outputs.csv"
DEFAULT_PER_QUERY_OUT = ROOT / "results" / "ablation_metrics_per_query.csv"
DEFAULT_SUMMARY_OUT = ROOT / "results" / "ablation_metrics_summary.csv"

for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

smoothing = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def bleu(ref: str, hyp: str) -> float:
    ref_tokens = nltk.word_tokenize(ref.lower())
    hyp_tokens = nltk.word_tokenize(hyp.lower())
    if not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


def rouge_l(ref: str, hyp: str) -> float:
    return rouge.score(ref, hyp)["rougeL"].fmeasure


def meteor(ref: str, hyp: str) -> float:
    ref_tokens = nltk.word_tokenize(ref.lower())
    hyp_tokens = nltk.word_tokenize(hyp.lower())
    if not hyp_tokens:
        return 0.0
    return meteor_score([ref_tokens], hyp_tokens)


def main(raw_path: Path, per_query_out: Path, summary_out: Path):
    df = pd.read_csv(raw_path)
    df["reference_answer"] = df["reference_answer"].fillna("")
    df["generated_answer"] = df["generated_answer"].fillna("")

    print(f"Scoring {len(df)} rows from {raw_path}...")

    # BLEU/ROUGE-L/METEOR computed from the ORIGINAL generated_answer values
    # (see below) -- a genuinely empty generated_answer must score exactly
    # 0.0 on all four metrics, not get silently rewarded for reference text
    # that happens to share a word with "[empty response]".
    df["bleu"] = [bleu(r, h) for r, h in zip(df["reference_answer"], df["generated_answer"])]
    df["rougeL"] = [rouge_l(r, h) for r, h in zip(df["reference_answer"], df["generated_answer"])]
    df["meteor"] = [meteor(r, h) for r, h in zip(df["reference_answer"], df["generated_answer"])]

    # A genuinely empty generated_answer (LLM returned "" without abstaining --
    # confirmed real, rare case, 2026-07-27, query Q071 in a reranker-ablation
    # run) crashes bert_score's tokenizer (RobertaTokenizer.
    # build_inputs_with_special_tokens([]) on a zero-token input), taking down
    # the whole batch rather than just that row. BLEU/METEOR already handle
    # this gracefully (return 0.0 for empty hyp_tokens, already scored above);
    # substituting a placeholder for BERTScore specifically avoids the crash
    # without touching the other three metrics' inputs. 2026-08-01 fix: this
    # substitution previously mutated the shared generated_answer column
    # BEFORE bleu/rougeL/meteor were computed from it, so those three were
    # silently scored against "[empty response]" instead of the true empty
    # string whenever the substitution fired -- inflating them away from the
    # correct 0.0 on any reference sharing a word with the placeholder text.
    # A local copy for BERTScore only, computed AFTER the three metrics
    # above already used the real (possibly empty) generated_answer values,
    # closes this.
    bertscore_input = df["generated_answer"].copy()
    n_empty = (bertscore_input.str.strip() == "").sum()
    if n_empty:
        print(f"WARNING: {n_empty} row(s) have an empty generated_answer -- "
              f"substituting a placeholder (BERTScore input only, not BLEU/ROUGE-L/METEOR) "
              f"so BERTScore doesn't crash on them.")
        bertscore_input.loc[bertscore_input.str.strip() == ""] = "[empty response]"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running BERTScore on {device} (downloads roberta-large on first call)...")
    P, R, F1 = bert_score_fn(
        bertscore_input.tolist(),
        df["reference_answer"].tolist(),
        lang="en",
        device=device,
        verbose=True,
        batch_size=16,  # bert-score's default batch size crashes (exit 5, no
        # traceback) on this bert-score/transformers version combo once the
        # batch is large enough (confirmed: 200 rows crashes unbatched, but
        # batch_size=16 processes all 200 cleanly) -- forcing a small batch
        # avoids it without changing the scores themselves.
    )
    df["bertscore"] = F1.tolist()

    per_query_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_query_out, index=False)
    print(f"Wrote per-query metrics to {per_query_out}")

    summary = df.groupby("config")[["bleu", "rougeL", "bertscore", "meteor"]].mean().round(4)
    summary["n"] = df.groupby("config").size()
    summary.to_csv(summary_out)
    print(f"Wrote summary metrics to {summary_out}")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--per-query-out", type=Path, default=DEFAULT_PER_QUERY_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()
    main(args.raw, args.per_query_out, args.summary_out)
