"""
Paired significance test: does re-fine-tuning on the tripled BanglishQA
table (1053->3044 rows, scripts/ingest_new_banglish_dataset.py) actually
improve Banglish-specific retrieval, or does adding much more code-mixed
training data risk degrading the base model the way a comparable published
system found for SOME base-model/fine-tuning combinations (InfoTextCM,
FIRE 2024 -- verified via literature search, 2026-07-31: fine-tuning
off-the-shelf SBERT models on code-mixed text sometimes DEGRADED
performance vs. zero-shot, base-model-dependent, not a guaranteed win)?

Compares finetuned_minilm_hard_negatives_structured_banglish_expanded
(trained on the expanded data) against the CURRENTLY DEPLOYED
finetuned_minilm_hard_negatives_structured (trained before the expansion)
on THREE held-out sets:
  1. Banglish-only (results/embedding_held_out_eval_banglish_only.csv's
     methodology, n=317 -- the direct question this ablation exists to
     answer)
  2. QA-pair overall (English+Banglish pooled, existing methodology) --
     does it regress the already-validated pooled metric?
  3. Structured-table (existing methodology) -- does it regress the
     already-fixed structured-table regression from earlier this project?

Usage: python scripts/compare_banglish_expanded_significance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_embeddings_held_out import load_val_pairs as load_qa_pairs
from eval_embeddings_held_out_banglish_only import load_banglish_val_pairs

MODEL_NEW = str(ROOT / "models" / "finetuned_minilm_hard_negatives_structured_banglish_expanded")
MODEL_DEPLOYED = str(ROOT / "models" / "finetuned_minilm_hard_negatives_structured")
N_BOOTSTRAP = 2000
SEED = 42


def per_pair_metrics(model_path, queries, chunks, dedup_correct=False):
    model = SentenceTransformer(model_path)
    q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    c_emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ c_emb.T
    n = len(queries)
    rr = np.empty(n)
    top1 = np.empty(n)
    for i in range(n):
        order = np.argsort(-sims[i])
        if dedup_correct:
            target = chunks[i]
            matching = {j for j in range(n) if chunks[j] == target}
        else:
            matching = {i}
        rank = next(r + 1 for r, idx in enumerate(order) if idx in matching)
        rr[i] = 1.0 / rank
        top1[i] = 1.0 if rank == 1 else 0.0
    return rr, top1


def bootstrap_ci_diff(a, b, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_approx = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), lo, hi, p_approx


def run_comparison(label, queries, chunks, dedup_correct):
    rr_new, top1_new = per_pair_metrics(MODEL_NEW, queries, chunks, dedup_correct)
    rr_dep, top1_dep = per_pair_metrics(MODEL_DEPLOYED, queries, chunks, dedup_correct)
    print(f"\n=== {label} (n={len(queries)}) ===")
    for m_label, a, b in [("MRR", rr_new, rr_dep), ("Top-1", top1_new, top1_dep)]:
        mean_diff, lo, hi, p = bootstrap_ci_diff(a, b)
        print(f"{m_label}: banglish_expanded={a.mean():.4f} deployed={b.mean():.4f} "
              f"diff={mean_diff:.4f} CI=[{lo:.4f},{hi:.4f}] p={p:.4f} significant={not (lo <= 0 <= hi)}")


def main():
    banglish_pairs = load_banglish_val_pairs()
    banglish_q = [q for q, a in banglish_pairs]
    banglish_a = [a for q, a in banglish_pairs]
    run_comparison("Banglish-ONLY held-out set (the direct question)", banglish_q, banglish_a, dedup_correct=False)

    qa_pairs = load_qa_pairs()
    qa_q = [q for q, a in qa_pairs]
    qa_a = [a for q, a in qa_pairs]
    run_comparison("QA-pair pooled held-out set (regression check)", qa_q, qa_a, dedup_correct=False)

    sdf = pd.read_csv(ROOT / "data" / "structured_qa_pairs.csv")
    sdf = sdf[sdf["split"] == "val"]
    struct_q = list(sdf["query"])
    struct_c = list(sdf["chunk_text"])
    run_comparison("Structured-table held-out set (regression check)", struct_q, struct_c, dedup_correct=True)


if __name__ == "__main__":
    main()
