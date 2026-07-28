"""
Fine-tune the retriever's embedding model (all-MiniLM-L6-v2) on our own
(question, answer) pairs via MultipleNegativesRankingLoss (in-batch
negatives), so the vector stream carries a domain-adapted semantic signal
instead of a generic one.

Why this matters: every round tonight found full_hybrid/bm25_only nearly
tied rather than hybrid decisively winning. Root cause (not a bug):
all-MiniLM-L6-v2 was never trained on this domain (course codes, university
FAQ phrasing, our specific paraphrase structure), so its semantic judgments
are comparatively weak next to BM25's precision on this corpus's exact-term-
heavy content -- blending a strong signal with an uncalibrated weaker one
caps how much the blend can beat the strong signal alone. Domain-adapting
the embedding model is the direct fix, not another retrieval-scoring patch.

STRICT leakage guardrail: only Split='train' rows from EnglishQA/BanglishQA
are used for training pairs. Split='val' is held out for a before/after
sanity check. Split='test' is NEVER touched here -- every evaluation round
(A-J) scores against it, and any leakage here would invalidate every
reported number tonight.

Usage: python scripts/finetune_embeddings.py
"""

import argparse
import sqlite3
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "knowledge_base.db"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = ROOT / "models" / "finetuned_minilm_domain"


def load_pairs(split: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT Question, Answer FROM EnglishQA WHERE Split=? "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND Question IS NOT NULL AND Answer IS NOT NULL",
        (split,),
    )
    pairs = [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    cur.execute(
        "SELECT QuestionBanglish, AnswerEnglish FROM BanglishQA WHERE Split=? "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND QuestionBanglish IS NOT NULL AND AnswerEnglish IS NOT NULL",
        (split,),
    )
    pairs += [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    conn.close()
    return pairs


def main(base_model: str, out_dir: Path, batch_size: int = 32):
    train_pairs = load_pairs("train")
    val_pairs = load_pairs("val")
    print(f"Base model: {base_model}")
    print(f"Train pairs (EnglishQA+BanglishQA, Split=train only): {len(train_pairs)}")
    print(f"Val pairs (Split=val, held out, not trained on): {len(val_pairs)}")

    train_dataset = Dataset.from_dict({
        "anchor": [q for q, a in train_pairs],
        "positive": [a for q, a in train_pairs],
    })

    model = SentenceTransformer(base_model)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        warmup_steps=0.1,
        save_strategy="no",
        logging_steps=20,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.base_model, args.out_dir, args.batch_size)
