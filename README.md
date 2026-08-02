# Hybrid RAG Academic Advising Chatbot (BRAC University)

Adaptive hybrid BM25 + dense-vector retrieval-augmented generation for
BRAC University academic advising, answering both English and **Banglish**
(Bengali–English code-mixed) student questions. Generation uses a locally
hosted Llama-3.1-8B served by [Ollama](https://ollama.com) over plain HTTP —
no hosted LLM API, no LangChain.

Supports the paper *"Adaptive Hybrid Retrieval with Confidence-Gated
Abstention for Bilingual Academic Advising"* (`paper/paper.tex`). Every
number in the paper is traceable to a script in `scripts/` and an output
file in `results/` (see `CLAUDE.md.md` for the full experiment log and the
project's zero-fabrication ground rules).

## How it works

```
query ──► HybridRetriever.retrieve_adaptive          pipeline/hybrid_retriever.py
          ├─ BM25 (rank-bm25, shared Banglish-aware tokenizer)
          ├─ dense vectors (ChromaDB, fine-tuned MiniLM)
          ├─ exact-match forcing for course codes / sections / faculty
          │  names / initials / aliases (+ unambiguous-match score ceiling)
          └─ routing: entity-heavy vs. open-ended (per-route lambda)
      ──► NovelPipeline.build_context                pipeline/novel_pipeline.py
          ├─ ambiguous-entity widening + clarification notice
          ├─ prerequisite-graph chain injection      pipeline/prerequisite_graph.py
          ├─ optional: reranker, query translation,  pipeline/reranker.py
          │  entity-normalization fallback,          pipeline/ollama_client.py
          │  faculty-room lookup                     pipeline/faculty_room_lookup.py
          ├─ confidence-ordered ("zig-zag") context assembly
          └─ calibrated per-route abstention gate    pipeline/abstention.py
      ──► generation (Ollama, deterministic decoding) pipeline/ollama_client.py
          └─ optional conformal claim back-off       pipeline/conformal_abstention.py
```

Component defaults reflect this project's own ablations, not guesses:
the cross-encoder reranker and query translation are **off** by default
(each measured as neutral-to-negative here), entity normalization is **on**
(validated on malformed-query ablation), and conformal back-off is **off**
until real human-labeled calibration data exists. Each default's evidence
is documented in the docstring where it is set.

## Repository layout

| Path | What it is |
|---|---|
| `pipeline/` | The evaluated system (retriever, orchestrator, graph, abstention, clients) |
| `scripts/` | One script per experiment/ablation; each writes its own file under `results/` |
| `data/` | Built corpus, BM25 pickle, test-query CSVs, alias table |
| `dataset/` | Source CSVs (QA datasets, faculty/course sheets) |
| `results/` | Raw outputs, metrics, significance tests, run metadata/logs |
| `tests/` | Regression tests (plain asserts, no framework required) |
| `paper/` | SCITEPRESS-format paper source |
| `legacy/` | Orphaned v1 prototype, kept for reference (see `legacy/README.md`) |
| `create_and_populate_db.py` | Builds `knowledge_base.db` from `dataset/` |

## Setup

Requires Python 3.10+ (uses modern type syntax), a running Ollama instance
with `llama3.1:8b` pulled, and (optionally) CUDA for the embedding models.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # see the torch +cu121 note inside
ollama pull llama3.1:8b
```

Build the knowledge base and both retrieval indexes, in order:

```bash
python create_and_populate_db.py            # dataset/*.csv -> knowledge_base.db
python scripts/build_corpus.py              # -> data/corpus.jsonl
python scripts/build_bm25_index.py          # -> data/bm25_corpus.pkl
python scripts/build_chroma_index.py        # -> chroma_db/  (first build only --
                                            #    use stage_chroma_index_rebuild.py +
                                            #    swap_chroma_staging.py for a LIVE index)
python scripts/build_question_embeddings_cache.py   # optional startup-cost cache
python scripts/calibrate_abstention.py      # -> results/abstention_threshold.json
```

## Run

```bash
python scripts/chat.py                      # interactive chat (plain hybrid retrieval)
python scripts/run_novel_pipeline.py --smoke   # full novel pipeline, 5-query smoke test
python scripts/demo_queries.py              # scripted demo queries
```

## Tests

No test framework needed (pytest also works):

```bash
python tests/test_patterns.py
python tests/test_dynamic_alpha.py
python tests/test_conformal_abstention.py
```

## Reproducibility notes

- All LLM calls use temperature 0 with a fixed seed; sampling scripts use
  fixed seeds (`SAMPLE_SEED = 42`).
- `requirements.txt` pins the direct dependencies verified in this
  project's own venv; `requirements_freeze_raw.txt` is the full freeze.
- Evaluation runs write a `*_metadata.json` beside their raw-output CSV
  (model, n, timestamps) so every results file is attributable.
- The fine-tuned embedding/reranker models under `models/` are not
  committed (multi-GB); the code falls back to the base HuggingFace models
  automatically, and `scripts/finetune_*.py` reproduce the fine-tuned ones.
