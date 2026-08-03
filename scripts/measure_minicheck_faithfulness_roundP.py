"""
A fourth, genuinely different independent faithfulness cross-check
signal, after three NLI model swaps (MiniLM, DeBERTa-base, DeBERTa
-large-FEVER: r=-0.300 -> -0.076 -> -0.018, a consistent but still-not
-positive trend). Every prior attempt was a generic natural-language
-inference model (trained on SNLI/MultiNLI/ANLI/etc., entailment as an
abstract relation between two arbitrary sentences) repurposed for
faithfulness checking via the SummaC-ZS sentence-decomposition trick.

MiniCheck (Tang, Laban & Durrett, EMNLP 2024, "MiniCheck: Efficient
Fact-Checking of LLMs on Grounding Documents") is not a repurposed NLI
model -- it is trained directly for "does this document support this
claim," the exact task this cross-check needs, with GPT-4-level accuracy
on grounding-verification benchmarks at a fraction of the cost (per the
paper's own claims, independently WebFetch-verified against the paper
and GitHub repo before use, not taken on trust). Uses lytang/MiniCheck-
DeBERTa-v3-Large (~0.4B params) via the official `minicheck` pip package
(installed from the maintainers' own GitHub repo -- a real, if new, trust
surface for this project: this is the first script here that adds a new
pip dependency rather than only downloading a model checkpoint through an
already-used library interface. Checked before installing: standard pip
install (not trust_remote_code), from the paper's own maintainers, EMNLP
2024 peer-reviewed).

Runs on the identical 169-row roundP sample used by all three prior NLI
checks, for a direct four-way comparison. Method: MiniCheck's own
`.score(docs, claims)` already handles document-level relevance/chunking
internally (unlike the manual SummaC-ZS context-sentence loop the other
three scripts use) -- per the library's own guidance, only the CLAIM side
needs sentence-level decomposition for best results, so each row's
generated_answer is split into sentences (nltk, same tokenizer already
used elsewhere in this project), scored against the full retrieved_context
as one document, and averaged across answer sentences.

Uses GPU automatically if available (MiniCheck's Inferencer uses HF
`device_map="auto"`, verified via WebFetch against the library source) --
important given a same-size DeBERTa-v3-large model already showed a
severe CPU performance anomaly in this project tonight (9+ hours, 8/38
batches) that resolved to 18 seconds on GPU; verified GPU is free
(no llama-server.exe / Ollama contention) before running.

Pure evaluation-instrument experiment. Does not touch pipeline/
generation/retrieval code, and does not replace the three prior NLI
scripts -- reported alongside them, whatever the result.

OUTCOME (2026-08-03): blocked before producing any result, for a
legitimate reason we did not work around. lytang/MiniCheck-DeBERTa-v3
-Large's only published checkpoint is a legacy pickle-format
`pytorch_model.bin` (1.74GB, no safetensors variant available in the
repo -- checked directly). Current `transformers` refuses to deserialize
non-safetensors checkpoints via `torch.load` unless torch>=2.6 (a
deliberate block for CVE-2025-32434, a real `torch.load` pickle
-deserialization vulnerability); this project's installed torch is
2.5.1. We did not bypass this check or upgrade torch to force it through
-- upgrading torch mid-session risks breaking every other already
-verified component's reproducibility (embeddings, reranker, generation),
and overriding a security check specifically protecting against
untrusted-pickle code execution is not a call to make casually just to
finish one more experiment. The `minicheck` package (and its pulled-in
`openai`/`distro`/`jiter`/`sniffio` dependencies, none of which this
project otherwise needs) was uninstalled afterward rather than left
unused in the environment. This script is kept as a documented, honest
record of a real attempt blocked by a genuine environment constraint --
runnable by someone with torch>=2.6 (and `pip install minicheck` re-run),
not by this project's current environment as configured.

Usage (requires torch>=2.6 and `pip install minicheck`, neither present
in this environment as of 2026-08-03): python scripts/measure_minicheck_faithfulness_roundP.py
"""

import sys
from pathlib import Path

import nltk
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from minicheck.minicheck import MiniCheck

INPUT_FILES = {
    "baselines": ROOT / "results" / "faithfulness_sample_baselines_roundP_faithfulness.csv",
    "novel": ROOT / "results" / "faithfulness_sample_novel_roundP_faithfulness.csv",
}
OUT_PER_QUERY = ROOT / "results" / "minicheck_faithfulness_per_query_roundP.csv"
OUT_SUMMARY = ROOT / "results" / "minicheck_faithfulness_summary_roundP.csv"
OUT_AGREEMENT = ROOT / "results" / "minicheck_vs_llmjudge_agreement_roundP.csv"


def main():
    print("Loading lytang/MiniCheck-DeBERTa-v3-Large (device_map=auto, GPU if available)...")
    scorer = MiniCheck(model_name="deberta-v3-large", cache_dir=str(ROOT / ".hf_cache"))

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

    all_docs, all_claims, row_ranges = [], [], []
    for idx, row in scoreable.iterrows():
        context = str(row["retrieved_context"])
        answer_sents = nltk.sent_tokenize(str(row["generated_answer"])) or [""]
        start = len(all_claims)
        for a_sent in answer_sents:
            all_docs.append(context)
            all_claims.append(a_sent)
        row_ranges.append((idx, start, len(answer_sents)))

    print(f"Scoring {len(all_claims)} (context, answer-sentence) claim pairs...")
    _, raw_probs, _, _ = scorer.score(docs=all_docs, claims=all_claims)

    row_scores = {}
    for idx, start, n in row_ranges:
        seg = raw_probs[start:start + n]
        row_scores[idx] = float(sum(seg) / len(seg)) if seg else 0.0
    scoreable["minicheck_faithfulness"] = scoreable.index.map(row_scores)

    OUT_PER_QUERY.parent.mkdir(parents=True, exist_ok=True)
    scoreable.to_csv(OUT_PER_QUERY, index=False)
    print(f"Wrote per-query MiniCheck faithfulness to {OUT_PER_QUERY}")

    group_col = "config" if "config" in scoreable.columns else "sample_source"
    summary = scoreable.groupby(group_col)["minicheck_faithfulness"].agg(["mean", "count"])
    summary.to_csv(OUT_SUMMARY)
    print(f"Wrote summary to {OUT_SUMMARY}")
    print(summary)

    if "faithfulness" in scoreable.columns:
        both = scoreable.dropna(subset=["faithfulness", "minicheck_faithfulness"])
        corr = both["faithfulness"].corr(both["minicheck_faithfulness"])
        agree = ((both["faithfulness"] >= 0.5) == (both["minicheck_faithfulness"] >= 0.5)).mean()
        agreement_row = pd.DataFrame([{
            "n_both_scored": len(both),
            "pearson_r_llmjudge_vs_minicheck": corr,
            "binary_agreement_at_0.5": agree,
        }])
        agreement_row.to_csv(OUT_AGREEMENT, index=False)
        print(f"\nLLM-judge vs MiniCheck agreement (n={len(both)}):")
        print(f"  Pearson r: {corr:.3f}")
        print(f"  Binary agreement (both >=0.5 or both <0.5): {agree:.3f}")
        print(f"Wrote {OUT_AGREEMENT}")


if __name__ == "__main__":
    main()
