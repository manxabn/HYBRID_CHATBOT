"""
Paired significance test: finetuned_minilm_hard_negatives (scripts/finetune_
embeddings_hard_negatives.py) vs. the currently-deployed finetuned_minilm_
domain (scripts/finetune_embeddings.py), on the same 297 held-out Split='val'
question/answer pairs scripts/eval_embeddings_held_out.py already evaluates
both on -- paired at the level of each pair's own reciprocal rank, not just
comparing the two models' aggregate means.

Usage: python scripts/compare_hard_negatives_significance.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sentence_transformers import SentenceTransformer
from eval_embeddings_held_out import load_val_pairs

MODEL_A = str(ROOT / "models" / "finetuned_minilm_hard_negatives")
MODEL_B = str(ROOT / "models" / "finetuned_minilm_domain")
N_BOOTSTRAP = 2000
SEED = 42


def per_pair_reciprocal_ranks(model_path, pairs):
    model = SentenceTransformer(model_path)
    questions = [q for q, a in pairs]
    answers = [a for q, a in pairs]
    q_emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    a_emb = model.encode(answers, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ a_emb.T
    n = len(pairs)
    rr = np.empty(n)
    top1 = np.empty(n)
    for i in range(n):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
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


def main():
    pairs = load_val_pairs()
    print(f"Held-out val pairs: {len(pairs)}")

    rr_a, top1_a = per_pair_reciprocal_ranks(MODEL_A, pairs)
    rr_b, top1_b = per_pair_reciprocal_ranks(MODEL_B, pairs)

    for label, a, b in [("MRR", rr_a, rr_b), ("Top-1", top1_a, top1_b)]:
        mean_diff, lo, hi, p = bootstrap_ci_diff(a, b)
        print(f"{label}: hard_negatives={a.mean():.4f} deployed={b.mean():.4f} "
              f"diff={mean_diff:.4f} CI=[{lo:.4f},{hi:.4f}] p={p:.4f} "
              f"significant={not (lo <= 0 <= hi)}")


if __name__ == "__main__":
    main()
