"""
Paired significance test: finetuned_minilm_hard_negatives_structured (QA
pairs + synthetic structured-table pairs) vs. the currently-deployed
finetuned_minilm_hard_negatives (QA pairs only), on BOTH held-out sets --
the QA-pair set (does extending training regress the already-validated
task?) and the new structured-table set (does it actually fix the
regression the original QA-pairs-only model showed there?). Paired at the
level of each row's own reciprocal rank/top-1 hit, same method as scripts/
compare_hard_negatives_significance.py.

Usage: python scripts/compare_structured_extension_significance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sentence_transformers import SentenceTransformer
from eval_embeddings_held_out import load_val_pairs as load_qa_pairs

MODEL_A = str(ROOT / "models" / "finetuned_minilm_hard_negatives_structured")
MODEL_B = str(ROOT / "models" / "finetuned_minilm_hard_negatives")
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
    rr_a, top1_a = per_pair_metrics(MODEL_A, queries, chunks, dedup_correct)
    rr_b, top1_b = per_pair_metrics(MODEL_B, queries, chunks, dedup_correct)
    print(f"\n=== {label} (n={len(queries)}) ===")
    for m_label, a, b in [("MRR", rr_a, rr_b), ("Top-1", top1_a, top1_b)]:
        mean_diff, lo, hi, p = bootstrap_ci_diff(a, b)
        print(f"{m_label}: structured={a.mean():.4f} qa_only={b.mean():.4f} "
              f"diff={mean_diff:.4f} CI=[{lo:.4f},{hi:.4f}] p={p:.4f} significant={not (lo <= 0 <= hi)}")


def main():
    qa_pairs = load_qa_pairs()
    qa_queries = [q for q, a in qa_pairs]
    qa_chunks = [a for q, a in qa_pairs]
    run_comparison("QA-pair held-out set (does extension regress this?)", qa_queries, qa_chunks, dedup_correct=False)

    sdf = pd.read_csv(ROOT / "data" / "structured_qa_pairs.csv")
    sdf = sdf[sdf["split"] == "val"]
    struct_queries = list(sdf["query"])
    struct_chunks = list(sdf["chunk_text"])
    run_comparison("Structured-table held-out set (does extension fix the regression?)",
                    struct_queries, struct_chunks, dedup_correct=True)


if __name__ == "__main__":
    main()
