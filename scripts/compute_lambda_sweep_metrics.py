"""
Score results/lambda_sweep_raw_outputs.csv against reference_answer, write
results/lambda_sweep_metrics.csv (mean per lambda, overall + entity-heavy vs
open-ended split), and plot metric-vs-lambda curves to
results/lambda_sweep_plot.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import torch
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "results" / "lambda_sweep_raw_outputs.csv"
METRICS_OUT = ROOT / "results" / "lambda_sweep_metrics.csv"
PLOT_OUT = ROOT / "results" / "lambda_sweep_plot.png"

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


def bleu(ref, hyp):
    r, h = nltk.word_tokenize(ref.lower()), nltk.word_tokenize(hyp.lower())
    return sentence_bleu([r], h, smoothing_function=smoothing) if h else 0.0


def rouge_l(ref, hyp):
    return rouge.score(ref, hyp)["rougeL"].fmeasure


def meteor(ref, hyp):
    r, h = nltk.word_tokenize(ref.lower()), nltk.word_tokenize(hyp.lower())
    return meteor_score([r], h) if h else 0.0


def main():
    df = pd.read_csv(RAW_PATH)
    df["reference_answer"] = df["reference_answer"].fillna("")
    df["generated_answer"] = df["generated_answer"].fillna("")

    df["bleu"] = [bleu(r, h) for r, h in zip(df["reference_answer"], df["generated_answer"])]
    df["rougeL"] = [rouge_l(r, h) for r, h in zip(df["reference_answer"], df["generated_answer"])]
    df["meteor"] = [meteor(r, h) for r, h in zip(df["reference_answer"], df["generated_answer"])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running BERTScore on {device}...")
    _, _, F1 = bert_score_fn(df["generated_answer"].tolist(), df["reference_answer"].tolist(),
                              lang="en", device=device, verbose=True)
    df["bertscore"] = F1.tolist()

    df.to_csv(ROOT / "results" / "lambda_sweep_metrics_per_query.csv", index=False)

    overall = df.groupby("lambda")[["bleu", "rougeL", "bertscore", "meteor"]].mean().round(4)
    overall["subset"] = "all"
    entity = df[df.is_entity_heavy].groupby("lambda")[["bleu", "rougeL", "bertscore", "meteor"]].mean().round(4)
    entity["subset"] = "entity_heavy"
    openq = df[~df.is_entity_heavy].groupby("lambda")[["bleu", "rougeL", "bertscore", "meteor"]].mean().round(4)
    openq["subset"] = "open_ended"

    summary = pd.concat([overall, entity, openq]).reset_index()
    summary.to_csv(METRICS_OUT, index=False)
    print(f"Wrote {METRICS_OUT}")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = ["bleu", "rougeL", "bertscore", "meteor"]
    for ax, metric in zip(axes.flat, metrics):
        for subset_name, marker in [("all", "o"), ("entity_heavy", "s"), ("open_ended", "^")]:
            sub = summary[summary.subset == subset_name]
            ax.plot(sub["lambda"], sub[metric], marker=marker, label=subset_name)
        ax.set_xlabel("lambda (0=vector-only, 1=BM25-only)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Metric vs. lambda (fixed 60-query subset, 30 entity-heavy + 30 open-ended)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    print(f"Wrote {PLOT_OUT}")


if __name__ == "__main__":
    main()
