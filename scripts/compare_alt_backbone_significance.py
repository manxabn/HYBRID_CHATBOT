"""
Paired significance test: does the hard-negative + structured-table
fine-tuning recipe's measured gain transfer to a DIFFERENT embedding model
family (intfloat/multilingual-e5-small), or is the MiniLM result specific to
that model? Compares base_e5small_pretrained vs. finetuned_e5small_hard_
negatives_structured on BOTH held-out sets, same paired-bootstrap method as
scripts/compare_structured_extension_significance.py (reuses its
per_pair_metrics/bootstrap_ci_diff directly, not a re-implementation).

Point-estimate context (results/embedding_held_out_eval.csv,
results/structured_embedding_held_out_eval.csv, both already measured,
2026-07-29): QA-pair set top1 0.192->0.279, MRR 0.405->0.520;
structured-table set top1 0.960->1.000, MRR 0.977->1.000 (E5's own BASE
model already scores far higher than base MiniLM's 0.646 on structured-table
matching -- its retrieval-oriented pretraining transfers well zero-shot --
so the fine-tuned ceiling there is a smaller absolute jump from an already-
high base, unlike MiniLM's 0.646->1.000).

Usage: python scripts/compare_alt_backbone_significance.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_embeddings_held_out import load_val_pairs as load_qa_pairs
from compare_structured_extension_significance import per_pair_metrics, bootstrap_ci_diff, run_comparison

MODEL_A_PATH = str(ROOT / "models" / "finetuned_e5small_hard_negatives_structured")
MODEL_B_PATH = "intfloat/multilingual-e5-small"


def run_comparison_e5(label, queries, chunks, dedup_correct):
    rr_a, top1_a = per_pair_metrics(MODEL_A_PATH, queries, chunks, dedup_correct)
    rr_b, top1_b = per_pair_metrics(MODEL_B_PATH, queries, chunks, dedup_correct)
    print(f"\n=== {label} (n={len(queries)}) ===")
    for m_label, a, b in [("MRR", rr_a, rr_b), ("Top-1", top1_a, top1_b)]:
        mean_diff, lo, hi, p = bootstrap_ci_diff(a, b)
        print(f"{m_label}: finetuned_e5={a.mean():.4f} base_e5={b.mean():.4f} "
              f"diff={mean_diff:.4f} CI=[{lo:.4f},{hi:.4f}] p={p:.4f} significant={not (lo <= 0 <= hi)}")


def main():
    qa_pairs = load_qa_pairs()
    qa_queries = [q for q, a in qa_pairs]
    qa_chunks = [a for q, a in qa_pairs]
    run_comparison_e5("QA-pair held-out set: fine-tuned E5-small vs. base E5-small",
                       qa_queries, qa_chunks, dedup_correct=False)

    sdf = pd.read_csv(ROOT / "data" / "structured_qa_pairs.csv")
    sdf = sdf[sdf["split"] == "val"]
    struct_queries = list(sdf["query"])
    struct_chunks = list(sdf["chunk_text"])
    run_comparison_e5("Structured-table held-out set: fine-tuned E5-small vs. base E5-small",
                       struct_queries, struct_chunks, dedup_correct=True)


if __name__ == "__main__":
    main()
