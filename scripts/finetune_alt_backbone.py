"""
Ablation: does the hard-negative + structured-table fine-tuning recipe
(scripts/finetune_embeddings_hard_negatives.py) transfer to a DIFFERENT
embedding model family, or is the measured gain specific to MiniLM? Every
embedding result in this project so far (base MiniLM, in-batch-only
fine-tune, hard-negative fine-tune, structured-extended fine-tune) uses one
model family -- a real, disclosed generalization-claim risk found during
this session's own codebase audit (confirmed via direct grep: every
embedding/reranker/NLI component uses a MiniLM variant).

Reuses scripts/finetune_embeddings_hard_negatives.py's main() directly
(same training recipe: QA pairs + synthetic structured-table pairs, mined
hard negatives) with a different base_model and a separate output
directory -- does not touch or overwrite the currently-deployed MiniLM
model.

Base model: intfloat/multilingual-e5-small (~118M params, still a "small"
model for fair comparison against MiniLM's ~22M params being not wildly
mismatched in compute budget, but a genuinely different pretraining
objective/architecture family -- E5's weakly-supervised contrastive
web-text pretraining vs. MiniLM's distilled-BERT lineage -- and natively
multilingual, relevant for this project's own bilingual setting).

Usage: python scripts/finetune_alt_backbone.py
"""

import os
import sys
from pathlib import Path

# Must be set BEFORE torch is imported (via sentence_transformers below) --
# PyTorch reads this env var when it first initializes its CUDA allocator,
# not on every call. This was the fix PyTorch's own OOM error message
# recommended verbatim for the SECOND E5-small crash (mining-step
# fragmentation) -- confirmed a NO-OP on this machine, though: the actual
# run logged "UserWarning: expandable_segments not supported on this
# platform" (Windows CUDA builds don't support it). Left set anyway
# (harmless, in case this ever runs on Linux) but the real fix for that
# crash was _encode_defragmented's periodic mid-loop torch.cuda.empty_cache()
# (scripts/finetune_embeddings_hard_negatives.py) -- confirmed effective,
# since mining then completed cleanly on the next two attempts. A THIRD
# crash (2026-07-29) hit the same "10.44 GiB allocated" signature but
# during actual TRAINING (50%, 204/408 steps) rather than mining -- fixed
# below via the same periodic-cache-clear idea applied to the training
# loop (train_empty_cache_every), plus gradient_checkpointing to reduce
# peak backward-pass memory directly, plus a smaller training batch_size.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_embeddings_hard_negatives import main as finetune_main

BASE_MODEL = "intfloat/multilingual-e5-small"
OUT_DIR = ROOT / "models" / "finetuned_e5small_hard_negatives_structured"

if __name__ == "__main__":
    # Stays on GPU (mine_device=None, i.e. auto-detect, same as every other
    # model in this project) per explicit instruction not to sidestep to
    # CPU. Third attempt's fixes, all additive on top of the second
    # attempt's mining fix (which is confirmed working -- mining has now
    # completed cleanly twice in a row):
    #   - batch_size 32->16 for TRAINING specifically (mine_batch_size was
    #     already 8): the crash moved from mining to training once mining
    #     was fixed, so training's own batch size gets the same "E5-small
    #     is ~5x bigger than MiniLM, the same batch size that's always
    #     been safe for MiniLM isn't" treatment.
    #   - train_empty_cache_every=10: periodic mid-training cache clearing
    #     (same idea as _encode_defragmented, applied to the training loop
    #     via a TrainerCallback -- HuggingFace's Trainer has no built-in
    #     equivalent).
    #   - gradient_checkpointing=True: directly reduces peak backward-pass
    #     memory (recomputes activations instead of storing them) --
    #     targets the OOM's actual crash site (a backward-pass linear
    #     layer) more directly than either fix above.
    finetune_main(base_model=BASE_MODEL, out_dir=OUT_DIR, batch_size=16, mine_batch_size=8,
                  train_empty_cache_every=10, gradient_checkpointing=True)
