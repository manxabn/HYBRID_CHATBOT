"""
Held-out validation of embedding-model quality: Top-1 accuracy, Top-5
accuracy, and MRR for question-to-answer retrieval among Split='val' rows
(EnglishQA + BanglishQA), which were held out from fine-tuning
(scripts/finetune_embeddings.py trains on Split='train' only). Never touches
Split='test'.

Written fresh rather than relying on recalled numbers from an earlier,
unsaved ad-hoc run this session -- this project's standing rule is that
every reported metric must trace to a file that was actually produced by
running code, not to memory.

Method: embed every val-split answer once (the "corpus" for this check),
embed every val-split question, and for each question rank all val answers
by cosine similarity -- is the question's OWN answer ranked first (Top-1),
in the top 5 (Top-5), and at what reciprocal rank (MRR)? This is a
paraphrase-retrieval sanity check, not the full hybrid pipeline.

Usage: python scripts/eval_embeddings_held_out.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "results" / "embedding_held_out_eval.csv"

MODELS = {
    "base_minilm_pretrained": "sentence-transformers/all-MiniLM-L6-v2",
    "finetuned_minilm_domain": str(ROOT / "models" / "finetuned_minilm_domain"),
    "finetuned_minilm_hard_negatives": str(ROOT / "models" / "finetuned_minilm_hard_negatives"),
    "finetuned_minilm_hard_negatives_structured": str(ROOT / "models" / "finetuned_minilm_hard_negatives_structured"),
    "finetuned_banglishbert_domain": str(ROOT / "models" / "finetuned_banglishbert_domain"),
    "finetuned_multilingual_minilm_domain": str(ROOT / "models" / "finetuned_multilingual_minilm_domain"),
}


def load_val_pairs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT Question, Answer FROM EnglishQA WHERE Split='val' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND Question IS NOT NULL AND Answer IS NOT NULL"
    )
    pairs = [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    cur.execute(
        "SELECT QuestionBanglish, AnswerEnglish FROM BanglishQA WHERE Split='val' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND QuestionBanglish IS NOT NULL AND AnswerEnglish IS NOT NULL"
    )
    pairs += [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    conn.close()
    return pairs


def evaluate(model_path: str, pairs):
    model = SentenceTransformer(model_path)
    questions = [q for q, a in pairs]
    answers = [a for q, a in pairs]
    q_emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    a_emb = model.encode(answers, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ a_emb.T  # [n_q, n_a]

    n = len(pairs)
    top1 = top5 = 0
    rr_sum = 0.0
    for i in range(n):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1  # 1-indexed
        top1 += rank == 1
        top5 += rank <= 5
        rr_sum += 1.0 / rank
    return {
        "top1_accuracy": top1 / n,
        "top5_accuracy": top5 / n,
        "mrr": rr_sum / n,
        "n": n,
    }


def main():
    pairs = load_val_pairs()
    print(f"Held-out val pairs (EnglishQA+BanglishQA, Split=val, never trained on): {len(pairs)}")

    rows = []
    for label, path in MODELS.items():
        if not Path(path).exists() and "/" not in path:
            print(f"Skipping {label}: {path} not found")
            continue
        print(f"Evaluating {label} ({path}) ...")
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
    print(out)


if __name__ == "__main__":
    main()
