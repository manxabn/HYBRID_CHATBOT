# Project Instructions — Hybrid RAG Academic Advising: Ablation Studies & Novelty

## What this project is
Codebase for an adaptive hybrid BM25 + ChromaDB retrieval-augmented generation
chatbot for BRAC University academic advising, handling both English and
Banglish (Bengali-English code-mixed) queries. Generation is a locally hosted
Llama-3.1-8B model served via Ollama, called over plain HTTP
(`pipeline/ollama_client.py`), not via LangChain or a hosted API. The paper
this supports is "Adaptive Hybrid Retrieval with Confidence-Gated Abstention
for Bilingual Academic Advising" (SCITEPRESS format, `paper/paper.tex`).

**This file was last brought current on 2026-07-28.** Earlier versions of
this file described the project at a much earlier, less mature stage
(fabricated ablation numbers, a fabricated 40-student survey, an unpopulated
alias table, RRF fusion implemented-but-unevaluated, a stale dataset). All of
that has since been resolved — see "Resolved since the last rewrite of this
file" below before assuming any of it is still an open problem.

## Non-negotiable ground rule (unchanged, still in force)
**Every number that goes into the paper must come from code that was actually
run, on the actual data/model, with the output saved to disk.**
Do not estimate, extrapolate, or "fill in a plausible value" for any metric.
If a run fails or an experiment can't be completed, say so explicitly rather
than approximating a result. Every reported number should be traceable to a
specific script, a specific output file, and a specific timestamp.
**Every literature citation must be independently verified (WebSearch,
correct authors/venue/arXiv ID) before it is added to the bibliography.**
Do not cite a paper from memory alone.

## Where the real system lives
- `pipeline/` — the actual evaluated system: `hybrid_retriever.py` (BM25 +
  dense fusion, exact-match/unambiguous-match mechanism, adaptive routing),
  `novel_pipeline.py` (orchestrates retrieval + graph augmentation + context
  assembly + abstention + generation), `prerequisite_graph.py`,
  `abstention.py`, `reranker.py`, `ollama_client.py`, `banglish_normalize.py`,
  `patterns.py` (shared regex, single source of truth for course-code
  matching), `chroma_embedding.py` (embedding-function wrapper, currently
  pointed at `models/finetuned_minilm_hard_negatives`).
- `scripts/` — one script per experiment/ablation, each writing its own
  output file(s) under `results/`. Script names are self-describing; see
  each script's own docstring for what it measures and how to run it.
- `create_and_populate_db.py` (project root) — builds `knowledge_base.db`
  from the `dataset/` CSVs. Still the real, current DB-population entry
  point despite living at the root rather than under `scripts/`.
- `embeddings.py` (project root) — `ChromaEmbeddingFunction`, a thin
  GPU-aware SentenceTransformer wrapper. Still imported by
  `pipeline/chroma_embedding.py`; not dead despite its root-level location.
  Has a large dead commented-out block at the top of the file (an earlier
  draft of the same class) that should eventually be deleted.

## Orphaned legacy code (do not build on this; not evaluated; not cited)
`main.py`, `agent.py`, `chat_interface.py`, `db_ingestion.py`,
`knowledgebase.py`, `test.py`, `test_model.py` form a self-contained, mutually
-referencing v1 prototype (a LangChain `ZERO_SHOT_REACT_DESCRIPTION` agent
over a single search tool, backed by a local `llama-2-7b-chat.Q4_K_M.gguf`
via `llama-cpp-python`, hardcoded to a path on an old desktop machine that no
longer applies to this checkout). Confirmed via direct import-graph search:
nothing in `pipeline/` or `scripts/` imports any of these six files, and none
of them import anything from `pipeline/`. They are not wired into the
evaluated system at all. `inspect_chroma.py` is a small standalone debug
utility (list/inspect a Chroma collection's contents) — harmless, not
imported by anything, not part of the evaluated system either.
**Recommendation:** archive or delete this set once the user confirms it's
safe to do so (not done automatically — this is the user's original
early-stage work, not something to remove without asking).

## Version control status: NONE
This directory is not a git repository, and git is not installed on this
machine as of 2026-07-28. Every edit to every file in this project — code,
data, and paper — has no history, no diff, and no rollback path other than
manual file copies. For a paper whose central methodological claim is
reproducibility and rigor, this is a real risk or the released code is not
reproducibely versioned. Recommend `git init` + an initial commit before
any further major changes, and before any code release alongside the paper;
this needs the user's go-ahead since it touches the whole project.

## Dependencies
`requirements.txt` was regenerated 2026-07-28 from the actual working
`.venv`'s `pip freeze` (see `requirements_freeze_raw.txt` for the full
transitive closure). The previous version of this file listed
`langchain==0.0.123`, `chromadb==0.3.26`, `llama-cpp-python`, `ctransformers`
— all from the orphaned v1 prototype above, none of them what the currently-
evaluated system uses. Regenerate `requirements_freeze_raw.txt` again (`pip
freeze` inside `.venv`) any time dependencies change, rather than hand-
editing version numbers.

## Resolved since the last rewrite of this file
- **The four-config baseline ablation is real**, not fabricated — extensively
  re-run and re-verified across this and prior sessions, most recently after
  deploying a new embedding model (see below). Every number in `paper.tex`
  traces to a `results/*.csv` file.
- **The fabricated 40-student survey has been removed from `paper.tex`.**
  Verified directly (searched for "survey", "participants", "respondents",
  "Likert", "questionnaire" — no trace remains). This was previously flagged
  as "being handled separately by Abir, not a Claude Code task" — confirmed
  resolved, not re-litigated.
- **Dataset swap to `BRACU_QA_Dataset_FINAL`/Banglish/`FacultyAvailability`
  is complete and is what every current result reflects** (2,297 EnglishQA +
  1,053 BanglishQA + 814 FacultyAvailability + 586 CourseDetails + 223
  FacultyList + 52 Coordinator + 34 Prerequisites = 5,059 total records).
- **`build_test_queries.py` correctly draws from `Split == 'test'` and
  excludes `Category == 'Out of Scope / Unanswerable'`** for the main
  ablation set — the two gaps an earlier version of this file flagged as
  unresolved are both fixed, verified by reading the script directly.
- **RRF fusion has been extensively evaluated**, not just implemented: it's
  the adaptive router's entity-heavy branch, isolated-tested against fixed
  baselines (found to add no measurable benefit over the exact-match
  mechanism alone — a real negative result, not a gap), and its own `k`
  constant swept (also no effect, for the same underlying reason).
- **Embedding fine-tuning now uses mined hard negatives**
  (`scripts/finetune_embeddings_hard_negatives.py`), not just in-batch
  negatives — a verified, statistically significant improvement over the
  original fine-tune (Top-1 +0.044, MRR +0.034, p<0.05 both), fully deployed
  (ChromaDB re-embedded and rebuilt, every retrieval-only finding in the
  paper re-verified against the rebuilt index).
- **FacultyAvailability coverage gap is closed** (schedule-and-day query
  routing fix), generalized into a standing table-coverage audit script
  (`scripts/audit_table_coverage.py`) that currently finds zero gaps.
- **Course-code alias table remains intentionally minimal** (only
  "nummeth"/"numerical methods" → CSE330, both sourced from this file's own
  historical limitations example) — this was never populated from real
  Facebook-group data because that raw source text isn't present in this
  repo. `paper.tex` describes it honestly as "a small alias table," not as
  broad coverage. Populating it further remains a real, open opportunity
  (see Novelty directions below), not a fabrication risk as currently
  described.

## Known still-open items (real, not resolved by more engineering alone)
1. **Human hallucination annotation** — the automated LLM-judge faithfulness
   score is real and code-executed, but is not a substitute for a second
   human rater. A disagreement-prioritized 50-item sample is already
   prepared (`results/human_annotation_sample.csv`, blank rating columns)
   for whoever does this. This is the one item on this list that
   fundamentally needs a person, not more code.
2. **Adaptive routing's own contribution has been isolated and found to add
   no measurable benefit** over a simpler fixed-lambda + exact-match
   configuration (replicated across two independently-trained embedding
   models) — this is a genuine, reportable negative result, not a bug to
   fix. Don't try to "rescue" this finding without new evidence; report it
   honestly, as the paper currently does.
3. **λ=0.25 (not the deployed 0.5) is the empirically best fixed fusion
   weight for open-ended queries**, replicated across two embedding models,
   surviving Bonferroni correction on the stronger of the two measurements.
   Flagged in the paper as a promising lead requiring a properly held-out
   re-tuning pass before deployment — evaluating and selecting on the same
   200-query test set used throughout the paper would be circular.
4. **The full 200-query, 4-config, generation-quality ablation table has not
   been regenerated under the new (hard-negative) embedding model** — a full
   regeneration is a 5.5-10 hour job (`scripts/run_ablation.py`'s own
   documented estimate). A stratified n=20 confirmatory sample was run
   instead and is honestly labeled as such in the paper; it reproduced the
   same relative ordering across all four metrics as the existing n=200
   table (measured under the previous embedding model).
5. **Entity-normalization LLM fallback is implemented but not validated by
   its own ablation** (`use_entity_normalization` flag, default off).
6. **RRF's `k` constant and adaptive routing's design cannot currently be
   tested under ambiguous entity resolution**, because every entity-heavy
   query in the 200-query test set happens to resolve to an unambiguous
   exact match. Building a deliberately-ambiguous entity-resolution test
   subset (shared course codes across cross-listed sections, ambiguous
   faculty-name fragments) would let RRF's actual contribution — as opposed
   to the exact-match bonus riding on top of it — finally be measured.

## Output discipline (unchanged)
- Every experiment gets its own script and its own output file (CSV/JSON)
  saved under `results/` — don't just print to console and lose it.
- When reporting a result back for the paper, always point to the specific
  file and row/line it came from.
- `results/v1_naive_bm25/` holds a deliberately-preserved prior-baseline
  snapshot from an earlier, pre-fix system state — keep treating any
  similarly superseded result set as a labeled historical baseline (a
  subdirectory, clearly named), not as something to silently overwrite or
  delete.
