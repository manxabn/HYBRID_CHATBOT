"""
Held-out validation of embedding-model quality on structured-table chunks
(data/structured_qa_pairs.csv, Split=val -- never trained on by scripts/
finetune_embeddings_hard_negatives.py when include_structured=True),
parallel to scripts/eval_embeddings_held_out.py's methodology for EnglishQA/
BanglishQA pairs but over the 5 structured tables instead: for each val
query, rank it against every val chunk_text (mixing across all 5 tables at
once, not per-table, since real retrieval never gets to assume which table
a query is about) and check whether its own chunk is ranked first (Top-1),
in the top 5 (Top-5), or its reciprocal rank (MRR).

Purpose: scripts/finetune_embeddings_hard_negatives.py's original (QA-pairs-
only) model showed a mixed result on full-corpus vector-only retrieval
(results/ir_metrics.csv) despite a clean win on the QA-pair-only held-out
set -- this script checks the specific hypothesis that motivated extending
training to include structured-table pairs: does the extended model do
BETTER on structured-table content specifically, without regressing on the
original QA-pair held-out set (still checked separately by eval_embeddings_
held_out.py, not duplicated here).

Usage: python scripts/eval_structured_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STRUCTURED_PAIRS_PATH = ROOT / "data" / "structured_qa_pairs.csv"
OUT_PATH = ROOT / "results" / "structured_embedding_held_out_eval.csv"

MODELS = {
    "base_minilm_pretrained": "sentence-transformers/all-MiniLM-L6-v2",
    "finetuned_minilm_hard_negatives (QA-pairs only, currently deployed)": str(ROOT / "models" / "finetuned_minilm_hard_negatives"),
    "finetuned_minilm_hard_negatives_structured (QA-pairs + structured)": str(ROOT / "models" / "finetuned_minilm_hard_negatives_structured"),
    "base_e5small_pretrained": "intfloat/multilingual-e5-small",
    "finetuned_e5small_hard_negatives_structured (alt-backbone ablation)": str(ROOT / "models" / "finetuned_e5small_hard_negatives_structured"),
}


def load_val_pairs():
    df = pd.read_csv(STRUCTURED_PAIRS_PATH)
    df = df[df["split"] == "val"]
    return list(zip(df["query"], df["chunk_text"]))


def evaluate(model_path: str, pairs):
    model = SentenceTransformer(model_path)
    queries = [q for q, _ in pairs]
    chunks = [c for _, c in pairs]
    q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    c_emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ c_emb.T

    n = len(pairs)
    top1 = top5 = 0
    rr_sum = 0.0
    for i in range(n):
        order = np.argsort(-sims[i])
        # Multiple queries can share the identical chunk_text (e.g. two
        # FacultyList queries about the same person both point at the same
        # doc) -- any rank whose chunk_text matches this query's OWN chunk
        # counts as correct, not just the literal index i, since that's the
        # real notion of "found the right answer" here.
        target_text = chunks[i]
        matching_positions = {j for j in range(n) if chunks[j] == target_text}
        rank = None
        for r, idx in enumerate(order):
            if idx in matching_positions:
                rank = r + 1
                break
        top1 += rank == 1
        top5 += rank <= 5
        rr_sum += 1.0 / rank
    return {"top1_accuracy": top1 / n, "top5_accuracy": top5 / n, "mrr": rr_sum / n, "n": n}


def main():
    pairs = load_val_pairs()
    print(f"Held-out structured-table val pairs: {len(pairs)}")

    rows = []
    for label, path in MODELS.items():
        if not Path(path).exists() and "/" not in path:
            print(f"Skipping {label}: {path} not found")
            continue
        print(f"Evaluating {label} ...")
        try:
            metrics = evaluate(path, pairs)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        metrics["model"] = label
        rows.append(metrics)
        print(f"  top1={metrics['top1_accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} mrr={metrics['mrr']:.3f}")

    out = pd.DataFrame(rows)[["model", "top1_accuracy", "top5_accuracy", "mrr", "n"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
