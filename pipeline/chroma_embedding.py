"""
Compatibility wrapper around the existing embeddings.ChromaEmbeddingFunction
for chromadb 1.x, which requires embedding functions to implement name(),
get_config(), and build_from_config() (the old class predates this and is
used elsewhere as-is, e.g. knowledgebase.py, so it's wrapped here rather
than modified).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chromadb import Documents, Embeddings, EmbeddingFunction
from embeddings import ChromaEmbeddingFunction

# Domain-fine-tuned via scripts/finetune_embeddings_hard_negatives.py
# (MultipleNegativesRankingLoss with an explicit mined hard negative per
# anchor, on top of the usual in-batch negatives -- see that script's
# docstring), extended 2026-07-28 to also train on synthetic (query, chunk)
# pairs from the 5 structured tables (scripts/build_structured_qa_pairs.py),
# not just EnglishQA/BanglishQA pairs. Motivation: the QA-pairs-only version
# of this model, deployed briefly earlier the same day, showed a real
# regression on structured-table content specifically -- held-out structured
# -table matching accuracy (results/structured_embedding_held_out_eval.csv)
# was 0.466 Top-1, WORSE than even the untouched base model's 0.646, since
# structured-table chunks (~42% of the corpus) were entirely absent from
# what that fine-tuning ever saw. This extended version fixes that
# regression (Top-1 0.466->1.000, MRR 0.524->1.000, both paired-bootstrap
# significant p<0.0005, n=328) while NOT regressing the original QA-pair
# held-out set (MRR 0.536->0.524, Top-1 0.290->0.263, neither significant,
# p=0.42/0.18, n=297) -- see scripts/compare_structured_extension_
# significance.py. The 1.000 structured-table score should not be read as
# "solved perfectly": that held-out set's own construction (2-3 synthetic
# queries per record, e.g. "office room"/"email"/"designation" all pointing
# at the same chunk) makes correct-chunk retrieval structurally easier than
# the QA-pair task's genuine free-text semantic matching, since it mostly
# reduces to preserving an exact proper-noun/course-code string overlap
# between query and chunk -- the real, generalizable finding is the
# DIRECTION and significance of the fix, not the literal ceiling value.
# Falls back to the generic base model automatically if the fine-tuned
# directory doesn't exist (e.g. a fresh clone that hasn't run the fine-
# tuning script yet).
_ROOT = Path(__file__).resolve().parent.parent
_FINETUNED_PATH = _ROOT / "models" / "finetuned_minilm_hard_negatives_structured"
DEFAULT_MODEL = str(_FINETUNED_PATH) if _FINETUNED_PATH.exists() else "sentence-transformers/all-MiniLM-L6-v2"


class Chroma1xEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._inner = ChromaEmbeddingFunction(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self._inner.embed_documents(list(input))

    def embed_query(self, input: str):
        """Single string -> single embedding vector (used directly by
        pipeline/hybrid_retriever.py, matching embeddings.py's original
        single-query convention -- NOT the batch-oriented chromadb Protocol
        signature, since chromadb itself never calls this here; queries are
        always issued via precomputed query_embeddings=[...])."""
        return self._inner.embed_query(input)

    @staticmethod
    def name() -> str:
        return "custom-sentence-transformers-minilm"

    def get_config(self):
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config):
        return Chroma1xEmbeddingFunction(config.get("model_name", DEFAULT_MODEL))

    @staticmethod
    def validate_config(config) -> None:
        return
