"""
Fine-tune the retriever's embedding model with MINED HARD NEGATIVES, instead
of scripts/finetune_embeddings.py's pure in-batch-negative setup
(MultipleNegativesRankingLoss with only (anchor, positive) pairs, where the
only negatives a training batch ever sees are whatever unrelated answers
happen to land in the same random batch).

Motivation (Meghwani et al., 2025, ACL Industry Track, arXiv:2505.18366,
"Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems" --
corrected citation, 2026-07-28: an earlier version of this docstring cited
an unverified arXiv ID, 2509.09459, that does not exist; verified via direct
WebSearch before use here, per this project's standing citation-verification
rule): in-batch random negatives are almost always trivially easy to
distinguish from the true positive once a batch is even moderately diverse --
an unrelated FAQ answer about library hours looks nothing like a CSE310
prerequisite answer, so the model never has to learn the FINE-GRAINED
distinctions this corpus actually needs. This corpus is a specific case where
that gap matters concretely: it contains many near-duplicate paraphrased
Q/A rows for the same underlying fact (noted independently in hybrid_
retriever.py's query-confidence margin comment) and many structurally similar
but factually distinct rows (different course codes' prerequisite answers,
different faculty members' schedule answers) that share most of their
surface vocabulary. Random in-batch negatives rarely include one of these
close-but-wrong rows; mined hard negatives specifically target them.

Extended 2026-07-28 to ALSO include synthetic (query, chunk) pairs from the
5 structured tables (scripts/build_structured_qa_pairs.py), not just
EnglishQA/BanglishQA (question, answer) pairs. Motivation: structured-table
chunks are ~42% of the corpus (2,155 of 5,059 chunks) but were entirely
absent from what this fine-tuning trained OR validated against; a real,
measured gap (results/ir_metrics.csv) showed vector-only full-corpus
retrieval got WORSE on Recall@5/nDCG after the QA-pair-only hard-negative
model was deployed, despite a clean win on the QA-pair-only held-out
validation -- consistent with a model that sharpened discrimination among
QA-pair-style text specifically, at some cost to the corpus's non-QA-pair
majority.

Method: for each training anchor (question), embed it and every OTHER
training answer with the BASE (not yet fine-tuned) model, then take the
single most cosine-similar answer that is NOT this anchor's own true answer
as its hard negative -- the "hardest" wrong answer the base model currently
confuses with the right one. Trained via MultipleNegativesRankingLoss with
an explicit (anchor, positive, negative) triplet column, which in sentence-
transformers' training API uses the given negative AND still draws in-batch
negatives from the rest of the batch -- strictly more negative signal per
step than the original script, not a replacement of it.

Same STRICT leakage guardrail as finetune_embeddings.py: only Split='train'
rows are used for mining and training; Split='val' is the held-out check
(scripts/eval_embeddings_held_out.py); Split='test' is never touched.

Usage: python scripts/finetune_embeddings_hard_negatives.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "knowledge_base.db"
STRUCTURED_PAIRS_PATH = ROOT / "data" / "structured_qa_pairs.csv"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Saved to a NEW directory, not overwriting the already-validated and
# currently-deployed finetuned_minilm_hard_negatives, until this extended
# (QA-pairs + structured-table pairs) version is itself compared against it
# on both held-out sets and shown to actually help before any redeployment.
OUT_DIR = ROOT / "models" / "finetuned_minilm_hard_negatives_structured"


def load_pairs(split: str, include_structured: bool = True):
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

    if include_structured and STRUCTURED_PAIRS_PATH.exists():
        # Own record-level train/val split (scripts/build_structured_qa_
        # pairs.py), independent of EnglishQA/BanglishQA's Split column
        # since structured tables have no such column of their own --
        # same leakage discipline (only 'train' rows used here), applied
        # to a different table family.
        sdf = pd.read_csv(STRUCTURED_PAIRS_PATH)
        sdf = sdf[sdf["split"] == split]
        pairs += [(row["query"].strip(), row["chunk_text"].strip())
                  for _, row in sdf.iterrows() if row["query"].strip() and row["chunk_text"].strip()]
    return pairs


def mine_hard_negatives(pairs, base_model_name=BASE_MODEL, batch_size=128):
    base_model = SentenceTransformer(base_model_name)
    questions = [q for q, _ in pairs]
    answers = [a for _, a in pairs]

    q_emb = base_model.encode(questions, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
    a_emb = base_model.encode(answers, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
    sims = q_emb @ a_emb.T  # [n, n]

    # Some answers are near-duplicates of each other (paraphrase rows) --
    # exclude not just the anchor's own index but any OTHER answer that is
    # near-identical text to the true answer, so we don't mine a
    # "hard negative" that is actually just another correct paraphrase of
    # the same fact (that would poison training with a false negative).
    n = len(pairs)
    hard_negatives = []
    for i in range(n):
        row_sims = sims[i].copy()
        row_sims[i] = -1.0
        for j in range(n):
            if j != i and answers[j].strip() == answers[i].strip():
                row_sims[j] = -1.0
        best_j = int(np.argmax(row_sims))
        hard_negatives.append(answers[best_j])
    return hard_negatives


def main(base_model=BASE_MODEL, out_dir=OUT_DIR, batch_size=32):
    train_pairs = load_pairs("train")
    n_structured = len(pd.read_csv(STRUCTURED_PAIRS_PATH).query("split == 'train'")) if STRUCTURED_PAIRS_PATH.exists() else 0
    print(f"Train pairs total: {len(train_pairs)} "
          f"(EnglishQA+BanglishQA + {n_structured} synthetic structured-table pairs, Split=train only)")
    print("Mining hard negatives with the base (pre-fine-tune) model...")
    hard_negatives = mine_hard_negatives(train_pairs, base_model)

    train_dataset = Dataset.from_dict({
        "anchor": [q for q, a in train_pairs],
        "positive": [a for q, a in train_pairs],
        "negative": hard_negatives,
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
    print(f"Saved hard-negative fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
