# Legacy prototype (not used by the evaluated system)

These files (`main.py`, `agent.py`, `chat_interface.py`, `db_ingestion.py`,
`knowledgebase.py`, `test.py`, `test_model.py`) are an early prototype: a
LangChain `ZERO_SHOT_REACT_DESCRIPTION` agent over a single search tool,
backed by a local `llama-2-7b-chat.Q4_K_M.gguf` model via `llama-cpp-python`,
hardcoded to a path on an old machine that no longer applies to this
checkout.

Confirmed via direct import-graph search (2026-07-28): nothing in
`pipeline/` or `scripts/` imports any of these files, and none of them
import anything from `pipeline/`. They are not wired into the system the
paper evaluates at all. Moved here rather than deleted, so the user's
original early-stage work isn't lost.

The actual, currently-evaluated system lives in `pipeline/` (the retrieval,
generation, abstention, reranking, and prerequisite-graph logic) and
`scripts/` (one script per experiment/ablation). `create_and_populate_db.py`
and `embeddings.py`, at the project root, are NOT legacy -- they're still
load-bearing (DB population and the embedding-function wrapper
respectively) despite living outside `pipeline/`/`scripts/`.
