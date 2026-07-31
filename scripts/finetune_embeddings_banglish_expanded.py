"""
Re-runs the existing hard-negative + structured-table fine-tuning recipe
(scripts/finetune_embeddings_hard_negatives.py, UNCHANGED) after BanglishQA
nearly tripled (1053->3044 rows, scripts/ingest_new_banglish_dataset.py) --
load_pairs() already queries BanglishQA WHERE Split='train' directly, so
simply re-running produces a model trained on the full expanded Banglish
set automatically, no code changes needed to the recipe itself.

Saved to a NEW directory, not overwriting the currently-deployed
finetuned_minilm_hard_negatives_structured, until this is compared against
it on held-out sets (Banglish-only, QA-pair overall, structured-table) and
shown to actually help -- same discipline as every other embedding
ablation in this project.

Usage: python scripts/finetune_embeddings_banglish_expanded.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_embeddings_hard_negatives import main as finetune_main

OUT_DIR = ROOT / "models" / "finetuned_minilm_hard_negatives_structured_banglish_expanded"

if __name__ == "__main__":
    finetune_main(out_dir=OUT_DIR)
