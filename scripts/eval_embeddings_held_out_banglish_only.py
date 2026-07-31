"""
Banglish-ONLY held-out embedding evaluation -- scripts/eval_embeddings_
held_out.py pools EnglishQA+BanglishQA val pairs into one aggregate metric,
which would dilute a Banglish-specific effect (especially now that
BanglishQA nearly tripled, 1053->3044 rows, via scripts/ingest_new_
banglish_dataset.py). This isolates BanglishQA's own Split='val' rows so
the "did retraining on much more Banglish data actually help Banglish
retrieval specifically" question has a direct answer, not one blended
with the much larger English-side signal.

Same method as eval_embeddings_held_out.py (question-to-answer retrieval
among val rows, Top-1/Top-5/MRR), same models dict convention -- just a
different, Banglish-only pair source.

Usage: python scripts/eval_embeddings_held_out_banglish_only.py
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
OUT_PATH = ROOT / "results" / "embedding_held_out_eval_banglish_only.csv"

MODELS = {
    "base_minilm_pretrained": "sentence-transformers/all-MiniLM-L6-v2",
    "finetuned_minilm_hard_negatives_structured (currently deployed)":
        str(ROOT / "models" / "finetuned_minilm_hard_negatives_structured"),
}


def load_banglish_val_pairs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT QuestionBanglish, AnswerEnglish FROM BanglishQA WHERE Split='val' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND QuestionBanglish IS NOT NULL AND AnswerEnglish IS NOT NULL"
    )
    pairs = [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    conn.close()
    return pairs


def evaluate(model_path: str, pairs):
    model = SentenceTransformer(model_path)
    questions = [q for q, a in pairs]
    answers = [a for q, a in pairs]
    q_emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    a_emb = model.encode(answers, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ a_emb.T

    n = len(pairs)
    top1 = top5 = 0
    rr_sum = 0.0
    for i in range(n):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
        top1 += rank == 1
        top5 += rank <= 5
        rr_sum += 1.0 / rank
    return {"top1_accuracy": top1 / n, "top5_accuracy": top5 / n, "mrr": rr_sum / n, "n": n}


def main():
    pairs = load_banglish_val_pairs()
    print(f"Banglish-only held-out val pairs (Split=val, never trained on): {len(pairs)}")

    rows = []
    for label, path in MODELS.items():
        if not Path(path).exists() and "/" not in path:
            print(f"Skipping {label}: {path} not found")
            continue
        print(f"Evaluating {label} ...")
        metrics = evaluate(path, pairs)
        metrics["model"] = label
        rows.append(metrics)
        print(f"  top1={metrics['top1_accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} mrr={metrics['mrr']:.3f}")

    out = pd.DataFrame(rows)[["model", "top1_accuracy", "top5_accuracy", "mrr", "n"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(out)


if __name__ == "__main__":
    main()
