"""
Fine-tune the cross-encoder reranker on this corpus's own retrieval output,
using hard negatives from the ACTUAL hybrid retriever rather than random
negatives -- directly targeting the failure mode the reranker ablation
found (results/significance_tests_novel_roundK.csv): the generic, non-
fine-tuned cross-encoder measurably *hurt* ranking quality once the
embeddings were already domain-fine-tuned, because its (query, doc)
relevance judgments, trained on MS MARCO web-search data, disagree with
this corpus's own notion of relevance. The direct fix, symmetric with
scripts/finetune_embeddings.py's fix for the embedding model, is to
fine-tune the reranker on this domain too rather than leave it generic.

STRICT leakage guardrail, same as finetune_embeddings.py: only Split='train'
question/answer pairs are used, and only to build (query, candidate, label)
triples -- retrieval itself is read-only against the existing (already-
deployed) index, never touching Split='val'/'test' query files.

Training data construction: for each train-split question, retrieve the
adaptive pipeline's actual top-10 candidates. The candidate whose corpus
metadata Question field exactly matches this question is the positive
(label=1.0); up to 3 other retrieved candidates are hard negatives
(label=0.0) -- real near-misses this retriever actually produces, not
random unrelated text, which is a substantially more informative training
signal for a reranker than random negatives would be.

Usage: python scripts/finetune_reranker.py
"""

import sqlite3
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever

DB_PATH = ROOT / "knowledge_base.db"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT_DIR = ROOT / "models" / "finetuned_reranker_domain"
MAX_HARD_NEGATIVES = 3


def load_train_pairs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT Question, Answer FROM EnglishQA WHERE Split='train' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND Question IS NOT NULL AND Answer IS NOT NULL"
    )
    pairs = [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    cur.execute(
        "SELECT QuestionBanglish, AnswerEnglish FROM BanglishQA WHERE Split='train' "
        "AND Category != 'Out of Scope / Unanswerable' "
        "AND QuestionBanglish IS NOT NULL AND AnswerEnglish IS NOT NULL"
    )
    pairs += [(q.strip(), a.strip()) for q, a in cur.fetchall() if q.strip() and a.strip()]
    conn.close()
    return pairs


def build_training_examples(retriever: HybridRetriever, pairs):
    queries, docs, labels = [], [], []
    n_no_positive_found = 0
    for i, (question, answer) in enumerate(pairs):
        results, _ = retriever.retrieve_adaptive(question, top_n=10)
        positive = None
        negatives = []
        for r in results:
            meta = r["metadata"]
            q_field = meta.get("Question") or meta.get("QuestionBanglish")
            if q_field and q_field.strip() == question and positive is None:
                positive = r["text"]
            else:
                negatives.append(r["text"])
        if positive is None:
            # The retriever didn't surface this pair's own chunk in its
            # top-10 at all -- can't build a positive example from it, skip
            # rather than fabricate one (e.g. by using the raw answer text,
            # which isn't what the reranker will see at inference time).
            n_no_positive_found += 1
            continue
        queries.append(question)
        docs.append(positive)
        labels.append(1.0)
        for neg in negatives[:MAX_HARD_NEGATIVES]:
            queries.append(question)
            docs.append(neg)
            labels.append(0.0)
        if (i + 1) % 200 == 0:
            print(f"  built examples for {i+1}/{len(pairs)} queries", flush=True)
    print(f"Skipped {n_no_positive_found}/{len(pairs)} pairs (own chunk not in retrieved top-10)")
    return queries, docs, labels


def main():
    pairs = load_train_pairs()
    print(f"Train pairs (EnglishQA+BanglishQA, Split=train only): {len(pairs)}")

    retriever = HybridRetriever()
    queries, docs, labels = build_training_examples(retriever, pairs)
    print(f"Built {len(labels)} (query, doc, label) training examples "
          f"({sum(labels)} positive, {len(labels) - sum(labels)} negative)")

    train_dataset = Dataset.from_dict({"query": queries, "doc": docs, "label": labels})

    model = CrossEncoder(BASE_MODEL)
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=32,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=50,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    print(f"Saved fine-tuned reranker to {OUT_DIR}")


if __name__ == "__main__":
    main()
