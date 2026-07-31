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


def _encode_defragmented(model, texts, batch_size, empty_cache_every=10):
    # Manually chunks encode() and forces torch.cuda.empty_cache() every
    # `empty_cache_every` chunks, instead of relying on sentence-transformers'
    # own internal batching loop (a black box that only returns control once
    # ALL chunks are done) plus a single cache-clear at the very end. Found
    # necessary (2026-07-29) after a second genuine CUDA OOM on E5-small hit
    # 50 minutes and 204/408 steps into mining, reporting "10.44 GiB
    # allocated by PyTorch" against this 4.00 GiB card alongside a degraded
    # 17.18s/it rate -- both point at CUDA allocator fragmentation building
    # up unchecked across hundreds of small variable-length-sequence batches
    # (E5-small's larger hidden size means more, larger cached blocks than
    # MiniLM ever produced), not a single allocation that's simply too big
    # for one batch_size choice to dodge. Releasing the cache periodically,
    # mid-loop, directly targets that buildup instead of hoping it never
    # accumulates. Harmless for the already-validated MiniLM path (that
    # model never came close to fragmenting this GPU's 4GB budget, so this
    # is just a few extra no-op empty_cache() calls there).
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    chunks = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        chunks.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size))
        batch_i = i // batch_size
        if batch_i % empty_cache_every == 0:
            print(f"    encoded {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
            if cuda_ok:
                torch.cuda.empty_cache()
    if cuda_ok:
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def mine_hard_negatives(pairs, base_model_name=BASE_MODEL, batch_size=128, device=None):
    # device=None preserves the exact prior behavior (SentenceTransformer's
    # own auto-detection) for the already-validated MiniLM path.
    #
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set by the caller,
    # e.g. scripts/finetune_alt_backbone.py, before this module's imports
    # trigger any CUDA init) is the fix PyTorch's own OOM error message
    # recommends for exactly the "large allocated total but 0 bytes free"
    # fragmentation symptom seen on E5-small's second crash -- it changes
    # the CUDA caching allocator to manage memory in resizable virtual
    # segments instead of fixed-size cached blocks, so fragmented holes from
    # many variable-length batches can actually be reused instead of forcing
    # a new, larger reservation each time.
    base_model = SentenceTransformer(base_model_name, device=device)
    questions = [q for q, _ in pairs]
    answers = [a for _, a in pairs]

    q_emb = _encode_defragmented(base_model, questions, batch_size)
    a_emb = _encode_defragmented(base_model, answers, batch_size)
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


def _make_cache_clear_callback(every_n_steps: int):
    # A THIRD genuine CUDA OOM (2026-07-29): mining now completes cleanly
    # (see _encode_defragmented above), but training itself crashed at 50%
    # (204/408 steps) on E5-small, same "10.44 GiB allocated by PyTorch"
    # fragmentation signature as the earlier mining crashes -- confirming
    # this isn't a mining-specific issue but the CUDA allocator fragmenting
    # across hundreds of steps generally, and HuggingFace's Trainer has no
    # built-in periodic cache-clearing of its own. Subclasses the real
    # transformers.TrainerCallback (imported lazily so this module has no
    # hard transformers-internals dependency at load time for callers that
    # never use this) and overrides only on_step_end -- the same mid-loop
    # -defragmentation idea as _encode_defragmented, applied to the
    # training loop instead of the encode loop. Opt-in (only constructed
    # when train_empty_cache_every > 0) so the already-validated MiniLM
    # path is completely unaffected.
    from transformers import TrainerCallback

    class PeriodicCacheClearCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % every_n_steps == 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            return control

    return PeriodicCacheClearCallback()


def main(base_model=BASE_MODEL, out_dir=OUT_DIR, batch_size=32, mine_batch_size=128, mine_device=None,
         train_empty_cache_every=0, gradient_checkpointing=False):
    # mine_batch_size defaults to 128 (mine_hard_negatives' own prior
    # default) so the already-validated MiniLM path's behavior is completely
    # unchanged -- added 2026-07-29 specifically for scripts/finetune_alt_
    # backbone.py's alternative-backbone ablation, which hit a real CUDA OOM
    # at batch_size=128 on this 4GB-VRAM GPU: E5-small is ~118M params vs
    # MiniLM's ~22M, and this encoding step's peak memory scales with both
    # model size and batch_size, so MiniLM's default was never a problem but
    # a ~5x larger model at the same batch size was. Passing a smaller
    # mine_batch_size for a larger base model is the safe fix -- this
    # parameter is opt-in (must be explicitly passed) precisely so it can't
    # silently change anything for existing callers.
    train_pairs = load_pairs("train")
    n_structured = len(pd.read_csv(STRUCTURED_PAIRS_PATH).query("split == 'train'")) if STRUCTURED_PAIRS_PATH.exists() else 0
    print(f"Train pairs total: {len(train_pairs)} "
          f"(EnglishQA+BanglishQA + {n_structured} synthetic structured-table pairs, Split=train only)")
    print(f"Mining hard negatives with the base (pre-fine-tune) model "
          f"(encode batch_size={mine_batch_size}, device={mine_device or 'auto'})...")
    hard_negatives = mine_hard_negatives(train_pairs, base_model, batch_size=mine_batch_size, device=mine_device)

    train_dataset = Dataset.from_dict({
        "anchor": [q for q, a in train_pairs],
        "positive": [a for q, a in train_pairs],
        "negative": hard_negatives,
    })

    model = SentenceTransformer(base_model)
    loss = MultipleNegativesRankingLoss(model)

    # gradient_checkpointing=True (opt-in, default False preserves the
    # already-validated MiniLM path exactly): trades recomputing forward
    # activations during the backward pass for not storing them, directly
    # reducing the peak memory the backward pass needs -- the exact phase
    # that crashed (see _make_cache_clear_callback docstring). Slower per
    # step, but this is a one-time fine-tuning job, not a latency-sensitive
    # path.
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        warmup_steps=0.1,
        save_strategy="no",
        logging_steps=20,
        gradient_checkpointing=gradient_checkpointing,
    )

    callbacks = [_make_cache_clear_callback(train_empty_cache_every)] if train_empty_cache_every > 0 else None
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=callbacks,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"Saved hard-negative fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
