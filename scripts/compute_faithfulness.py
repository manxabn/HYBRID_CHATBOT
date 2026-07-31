"""
RAGAS-style Faithfulness metric: is the generated answer actually supported
by the retrieved context, or does it state things the context doesn't back
up? This is the real, automated, non-fabricated complement to the still-
fabricated Hallucination Annotation table in paper.tex -- same underlying
question (is this grounded or hallucinated), answered honestly with code
that actually ran, not invented numbers.

Why this, not BLEU/ROUGE/BERTScore: those measure similarity to a reference
answer, which can't distinguish a faithful paraphrase from a confidently
stated fabrication -- exactly the gap the 2025-2026 RAG evaluation
literature (RAGAS, ARES) identifies BLEU/ROUGE as unable to close (see
literature review, 2026-07-27).

Method (LLM-as-judge, using the SAME already-available local model rather
than a new dependency): for each row, ask the model whether the generated
answer's claims are supported by the retrieved context, and to output a
0-1 score plus a one-line justification. Rows with no retrieved context
(no_retrieval config) or an abstained/empty answer are skipped -- there's
nothing to check faithfulness against, or nothing to check.

Usage: python scripts/compute_faithfulness.py --raw results/ablation_raw_outputs_roundK.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import MODEL, OLLAMA_URL, post_with_retry
import requests

FAITHFULNESS_PROMPT = (
    "You are checking whether a generated ANSWER is fully supported by the "
    "retrieved CONTEXT it was supposed to be grounded in. Decompose the "
    "ANSWER into its individual factual claims, and for each claim check "
    "whether the CONTEXT actually supports it (not whether it's true in "
    "general, only whether the CONTEXT backs it up).\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Output EXACTLY one line in this format, nothing else:\n"
    "SCORE: <a number from 0.0 to 1.0, the fraction of claims supported by "
    "the context> | REASON: <one short sentence>"
)

SCORE_RE = re.compile(r"SCORE:\s*([0-9.]+)")


def judge_faithfulness(context: str, answer: str, timeout: int = 300) -> tuple[float | None, str]:
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 2048},
        },
        timeout=timeout,
    )
    text = resp.json()["response"].strip()
    m = SCORE_RE.search(text)
    score = float(m.group(1)) if m else None
    if score is not None:
        score = max(0.0, min(1.0, score))
    return score, text


def main(raw_path: Path, per_query_out: Path, summary_out: Path):
    df = pd.read_csv(raw_path)
    df["reference_answer"] = df.get("reference_answer", "").fillna("")
    df["generated_answer"] = df["generated_answer"].fillna("")
    context_col = "retrieved_context" if "retrieved_context" in df.columns else None

    scores, reasons = [], []
    n_skipped = 0
    for i, row in df.iterrows():
        context = row.get(context_col, "") if context_col else ""
        answer = row["generated_answer"]
        abstained = str(row.get("abstained", "")).strip().lower() == "true"
        if abstained or not str(context).strip() or not str(answer).strip():
            scores.append(None)
            reasons.append("skipped: no context, no answer, or abstained")
            n_skipped += 1
            continue
        print(f"  [{i+1}/{len(df)}] scoring {row.get('query_id', i)}...", flush=True)
        score, reason = judge_faithfulness(context, answer)
        scores.append(score)
        reasons.append(reason)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} scored ({n_skipped} skipped so far)")

    df["faithfulness"] = scores
    df["faithfulness_reason"] = reasons

    per_query_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_query_out, index=False)
    print(f"Wrote per-query faithfulness to {per_query_out}")

    scored = df[df["faithfulness"].notna()]
    if "config" in df.columns:
        summary = scored.groupby("config")["faithfulness"].agg(["mean", "count"])
    else:
        summary = pd.DataFrame({"mean": [scored["faithfulness"].mean()], "count": [len(scored)]})
    summary.to_csv(summary_out)
    print(f"Wrote summary faithfulness to {summary_out}")
    print(summary)
    print(f"Skipped {n_skipped}/{len(df)} rows (no context / no answer / abstained)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--per-query-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    per_query_out = args.per_query_out or (args.raw.parent / f"{args.raw.stem}_faithfulness.csv")
    summary_out = args.summary_out or (args.raw.parent / f"{args.raw.stem}_faithfulness_summary.csv")
    main(args.raw, per_query_out, summary_out)
