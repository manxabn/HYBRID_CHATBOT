# Project Instructions — Hybrid RAG Academic Advising: Ablation Studies & Novelty

## What this project is
Codebase for an adaptive hybrid BM25 + ChromaDB retrieval-augmented generation
chatbot for BRAC University academic advising, handling both English and
Banglish (Bengali-English code-mixed) queries. Generation is a locally hosted
Llama-3.1-8B model served via Ollama, called over plain HTTP
(`pipeline/ollama_client.py`), not via LangChain or a hosted API. The paper
this supports is "Adaptive Hybrid Retrieval with Confidence-Gated Abstention
for Bilingual Academic Advising" (SCITEPRESS format, `paper/paper.tex`).

**This file was last brought current on 2026-07-29.** A large round of code
-level fixes and additions happened since the previous rewrite (2026-07-28)
— see "Resolved/added since the last rewrite" below. `paper.tex` itself was
NOT touched during this round (explicit user instruction: fix code, not the
paper, during this phase) — none of the items below are reflected in the
paper yet.

## Non-negotiable ground rule (unchanged, still in force)
**Every number that goes into the paper must come from code that was actually
run, on the actual data/model, with the output saved to disk.**
Do not estimate, extrapolate, or "fill in a plausible value" for any metric.
If a run fails or an experiment can't be completed, say so explicitly rather
than approximating a result. Every reported number should be traceable to a
specific script, a specific output file, and a specific timestamp.
**Every literature citation must be independently verified (WebSearch,
correct authors/venue/arXiv ID) before it is added to the bibliography.**
Do not cite a paper from memory alone. This rule was tested hard this round:
an external document proposing ~20 "literature-grounded solutions" was
independently verified item-by-item, and 2 citations were found completely
fabricated (a nonexistent "2026 clustering hard-negative mining paper"; a
"2025 survey" claim about reranker recall@50 thresholds that, when actually
fetched, contained no such claim) and 5 more were real papers with invented
specifics attached (a fake "ensemble" finding attributed to a real paper
that found the opposite; a fabricated system name "FIT-Advisor" attached to
a real, differently-named paper; among others). Treat any externally
-supplied citation the same way — verify before use, every time.

## Where the real system lives
- `pipeline/` — the actual evaluated system: `hybrid_retriever.py` (BM25 +
  dense fusion, exact-match/unambiguous-match mechanism, adaptive routing,
  token-level faculty-name matching, ambiguous-entity detection),
  `novel_pipeline.py` (orchestrates retrieval + graph augmentation + context
  assembly + abstention + generation + conformal back-off), `prerequisite_
  graph.py`, `abstention.py`, `conformal_abstention.py` (new), `reranker.py`,
  `ollama_client.py`, `banglish_normalize.py`, `patterns.py` (shared regex),
  `chroma_embedding.py` (embedding-function wrapper, currently pointed at
  `models/finetuned_minilm_hard_negatives_structured`).
- `scripts/` — one script per experiment/ablation, each writing its own
  output file(s) under `results/`. Script names are self-describing.
- `create_and_populate_db.py` (project root) — builds `knowledge_base.db`
  from the `dataset/` CSVs. Still the real DB-population entry point.
- `embeddings.py` (project root) — `ChromaEmbeddingFunction`, still imported
  by `pipeline/chroma_embedding.py`. Dead commented-out block at the top was
  removed this round.
- `legacy/` — the orphaned v1 prototype (`main.py`, `agent.py`,
  `chat_interface.py`, `db_ingestion.py`, `knowledgebase.py`, `test.py`,
  `test_model.py`), moved here (not deleted) with the user's explicit
  confirmation. See `legacy/README.md`.

## Version control: NOW INITIALIZED
Git was installed (via winget) and the repository initialized this round,
with the user's explicit confirmation. Initial commit `458b6bb`, 561 files,
`.gitignore` extended to exclude `.venv/`, `models/`, `chroma_db/`,
`.hf_cache/`, `.nltk_data/`, `.pip_cache/`, `.mplconfig/`, `.mpl_cache/`,
`.tmp/`, `.torch_home/` (several multi-GB caches that weren't excluded
before and would have bloated the repo badly). Commit going forward as
normal; this is no longer a standing gap.

## Resolved/added since the last rewrite (2026-07-29 round)
- **Possessive-name normalization bug fixed**: `"Kaykobad's"` was being
  mangled to `"kaykobads"` (apostrophe deleted, not spaced) by
  `_normalize_name`, so possessive faculty-name queries matched nothing.
- **Token-level faculty-name matching added**: partial names (e.g. "Dr.
  Kaykobad" instead of the full stored "Dr. Mohammad Kaykobad") now resolve
  via `faculty_name_token_index`, not just full-name substring match.
- **Ambiguous multi-entity resolution, found and fixed**: a query like "What
  is Rahman's office room?" (16 real distinct faculty share that surname)
  used to silently retrieve only 5 arbitrary candidates and answer with
  false confidence naming one. Now widens retrieval to capture all true
  candidates (capped at `AMBIGUOUS_ENTITY_MAX=20`) and injects an explicit
  notice instructing the LLM to enumerate candidates and ask for
  clarification. Extended further this round with a **conditioning
  -attribute hint** (`build_conditioning_hint`): if the candidates differ on
  Designation/Initial/Status, that field and its values are surfaced too, so
  the user can disambiguate with a short attribute ("the professor") instead
  of re-typing a full name. Verified live on the Rahman case: real
  Designation values (Professor, Senior Lecturer, Lecturer, C. Lecturer, C.
  Senior Lecturer) now appear in context.
- **Entity-normalization fallback fully validated and fixed** (was
  previously "implemented, not validated" — see old item 5 below, now
  resolved): ablation went from 1/8 to 8/8 hits across three separate fixes:
  (a) the gating heuristic (`looks_like_unrecognized_entity`) could only
  ever fire for course-code-shaped (letters+digits) queries, never for a
  pure misspelled name — added `looks_like_unrecognized_name` as a second,
  independent gate; (b) the LLM's own generative spelling correction was
  unreliable on harder misspellings ("Shatobdo"→"Shatordo", still wrong) —
  replaced with deterministic `difflib` fuzzy matching against the closed
  223-person faculty roster (`fuzzy_correct_name`), tried before the LLM
  fallback; (c) two real control-flow bugs found and fixed: the re-retrieval
  -and-union step was nested inside only one `if/elif` branch so the fuzzy
  -corrected path never actually re-retrieved; and `generate_fn()` was
  called with the ORIGINAL misspelled query even when retrieval had
  correctly resolved the corrected name, causing the LLM to see a name
  mismatch between question and context and refuse to answer despite having
  the right information. `use_entity_normalization` now defaults to `True`.
- **Embedding fine-tuning extended to structured-table content**: the
  deployed hard-negative model was trained/validated only on EnglishQA/
  BanglishQA (question, answer) pairs — ~42% of the corpus (structured
  tables: CourseDetails, FacultyList, Coordinator, Prerequisites,
  FacultyAvailability) was entirely unrepresented. Built 2,155 synthetic
  (query, chunk) pairs from those 5 tables (`scripts/build_structured_qa_
  pairs.py`, proper record-level train/val split), retrained
  (`models/finetuned_minilm_hard_negatives_structured`), and confirmed via
  paired significance testing: fixes a real regression on structured-table
  held-out matching (Top-1 0.466→1.000, p<0.0005) with NO significant
  regression on the original QA-pair held-out set (p=0.42/0.18). Deployed:
  ChromaDB reindexed, full-corpus vector-only Recall@1 went from 0.560 to
  0.890 (nDCG@5 0.574→0.831); bm25_only/full_hybrid/adaptive unchanged, as
  expected since they're dominated by exact-match.
- **Conformal-style claim-decomposition back-off implemented**
  (`pipeline/conformal_abstention.py`), structurally following Mohri &
  Hashimoto's conformal factuality (ICML 2024, independently verified).
  Reuses the project's own existing SummaC-ZS-style NLI entailment scoring.
  **Route-aware, and this matters**: initial testing found a uniform
  threshold caused false abstention on completely correct entity-heavy
  answers (e.g. "Dr. Kaykobad's office room is 4G11" scored NLI=0.099
  despite being exactly right) — checking this project's own already
  -measured `nli_faithfulness_per_query.csv` confirmed this is systematic,
  not a fluke: entity_heavy-route rows score mean=0.251/median=0.106 vs.
  open_ended mean=0.744/median=0.958, a structural NLI-model blind spot for
  field:value context, not a real faithfulness difference. Fixed by
  deferring entirely to `exact_match_any` for entity_heavy queries and
  applying claim-level NLI scoring only to open_ended ones. Even there, a
  second check found several open_ended answers the LLM-judge scored fully
  faithful (1.0) still get NLI scores of 0.01-0.15 — meaning even the
  restricted mechanism isn't reliable enough to trust without real human
  -labeled calibration data (which doesn't exist yet, same standing gap as
  item 1 below). `use_conformal_backoff` defaults to `False` for exactly
  this reason; `calibrate_threshold_from_labels()` is ready for when
  calibration data exists (`results/human_annotation_sample.csv` is the
  natural candidate once labeled).
- **Post-hoc statistical power computed for this project's small-n
  ablations** (`scripts/power_analysis.py`, via scipy's noncentral
  t-distribution, no new dependency added): graph augmentation (n=12)
  achieved power 0.376, needs n≈32 for 80% power at its observed effect
  size; cross-lingual stress test original (n=9) achieved power 0.603,
  needs n≈14; expanded set (n=27, noisier variants) achieved power 0.369,
  needs n≈77. None of these are wrong to report, but none should be read
  as more settled than n this small actually supports.
- **Reranker failure mechanistically confirmed with new data**
  (`scripts/measure_prerank_pool_recall.py`): only 2/200 test queries (1%)
  have their correct chunk present in the reranker's candidate pool but not
  already ranked first — i.e. 91% of queries are already correct before
  reranking runs, so the reranker's theoretical ceiling of improvement is
  capped at 1% of the test set while its downside (disturbing an
  already-correct ranking) is unbounded. Real, own-project data explaining
  the already-established negative reranker result; not dependent on any
  external citation (an external "reranker needs recall@50<0.85" citation
  was checked and found fabricated — see above).
- **Governance-category dataset augmentation proposed** (`scripts/augment_
  governance_category.py` → `data/governance_augmentation_proposed.csv`,
  88 rows: 49 train/20 val/19 test). EnglishQA's smallest category,
  "University Facts & Governance" (66 rows, 22 unique facts), had only 2/22
  facts represented in val and 3/22 in test. Generates NEW PARAPHRASES of
  the 22 already-verified facts (does not invent new facts about the
  university), with an embedding-similarity diversity filter (reject any
  candidate too similar to an existing/already-accepted phrasing) so
  generated rows don't just collapse into near-duplicates. **PROPOSED
  ONLY** — `knowledge_base.db` is untouched; this needs the user's review
  before merging, same as every other higher-blast-radius dataset/DB
  decision this project treats as requiring explicit sign-off.
- **Alias table population attempted, correctly blocked, not forced**:
  checked whether course titles/subject names exist anywhere in this
  corpus (CourseDetails, Prerequisites, Coordinator, EnglishQA) before
  attempting LLM-assisted alias generation — they do not; only bare codes
  (e.g. "CSE330") exist anywhere, no "Numerical Methods"-style titles. An
  LLM generating informal-name aliases without that ground truth would be
  inventing code-to-subject mappings for a specific university's curriculum
  with no way to verify them — exactly the hallucination risk this
  project's standing rule forbids. Left undone rather than faked; genuinely
  requires new source data (real informal usage text, or an official
  course-title catalog) that isn't present in this repo.
- **Alternative embedding backbone ablation (E5-small)**: testing whether
  the hard-negative + structured-table fine-tuning recipe's gains are
  MiniLM-specific or transfer to a different model family (every embedding/
  reranker/NLI component in this project uses a MiniLM variant — a real,
  flagged generalization-claim risk). Status as of this file's last edit:
  [update this line once `scripts/finetune_alt_backbone.py` completes —
  check `results/` for `finetuned_e5small_hard_negatives_structured` eval
  outputs, or the absence of them, before assuming a result exists].

## 2026-07-29 overnight continuation (autonomous, user asleep)
User's explicit standing instruction for this round: keep researching and
fixing all night, "do not lose anything, do not harm any work to improve."
Everything below is additive/opt-in — nothing already-deployed was changed
in a way that alters default behavior, and every existing test still passes.
1. **Real bug found and fixed via full-codebase line audit**:
   `COURSE_CODE_RE` (`pipeline/patterns.py`) had no leading word-boundary
   and no check that its 3-digit group wasn't a truncated prefix of a
   longer number — confirmed live on production test-query CSVs: "...after
   29 June 2025?" and "...Wishlist event for Summer 2025?" were both
   parsed as containing a fake course code ("JUNE202"/"MMER202"), wrongly
   routing ordinary date-mentioning queries through the entity_heavy/RRF
   branch. Fixed with `\b` + `(?!\d)`; verified against all 5 test-query
   CSVs (date false positives gone, all 120+ real course-code matches
   unaffected); added a regression test (`tests/test_patterns.py`).
2. **Second instance of the "duplicated regex silently drifts" bug class**
   (the same class patterns.py's own docstring already names once) found in
   `scripts/calibrate_abstention.py`, which had its own local, unsynced
   copy of a course-code regex — fixed to import the shared one. Swept the
   rest of the codebase for the same pattern; nothing else found (one
   look-alike in `scripts/build_test_queries.py` is intentionally separate
   and documented as such — a diagnostic-only regex, not a bug).
3. **E5-small alt-backbone ablation: fixed a SECOND, different real CUDA
   OOM.** First crash (batch_size=128, mid-mining) was fixed with
   `mine_batch_size=16` — but a retry crashed again 50 minutes/204-408
   steps into mining, reporting "10.44 GiB allocated by PyTorch" against
   the actual 4GB card plus a degraded 17s/it rate: allocator fragmentation
   building up over hundreds of batches, not one oversized allocation.
   Fixed via manually-chunked encoding with `torch.cuda.empty_cache()`
   every 10 batches (`_encode_defragmented`,
   `scripts/finetune_embeddings_hard_negatives.py`) plus
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (though this specific
   flag turned out to be a no-op on Windows CUDA per its own warning — the
   periodic cache-clearing is what's actually working). Third attempt
   (task `bjaezdoj1`) got past mining cleanly for the first time — confirmed
   the mining fix works — but then crashed a THIRD time, during actual
   gradient training at 50% (204/408 steps), same "10.44 GiB allocated by
   PyTorch" fragmentation signature, just in the backward pass instead of
   the encode loop. Fourth attempt (task `bqbc3p2oc`) adds: training
   batch_size 32->16, a `TrainerCallback` that clears the CUDA cache every
   10 training steps (`_make_cache_clear_callback`, mirrors
   `_encode_defragmented`'s mid-loop clearing but for the training loop,
   which has no built-in equivalent), and `gradient_checkpointing=True`
   (trades compute for directly reduced backward-pass peak memory --
   targets the actual crash site). **Fourth attempt succeeded** — full 3
   epochs in 465s (train_loss 0.1866), saved to `models/finetuned_e5small_
   hard_negatives_structured`. Held-out eval (`results/embedding_held_out_
   eval.csv`, `results/structured_embedding_held_out_eval.csv`) plus a new
   paired-bootstrap significance script (`scripts/compare_alt_backbone_
   significance.py`, 2000 resamples) confirm the recipe TRANSFERS to a
   different model family, real result:
   - QA-pair held-out: Top-1 0.192->0.279 (diff=+0.088, p=0.001), MRR
     0.405->0.520 (diff=+0.116, p<0.0005). Both significant.
   - Structured-table held-out: Top-1 0.960->1.000 (diff=+0.040,
     p<0.0005), MRR 0.977->1.000 (diff=+0.022, p<0.0005). Both
     significant, though the absolute jump is smaller than MiniLM's
     0.646->1.000 because E5-small's BASE (untuned) model already scores
     far higher on structured-table matching than base MiniLM does — its
     retrieval-oriented pretraining transfers well zero-shot, a real and
     separately interesting finding in its own right, not just a
     ceiling-effect footnote.
   Not deployed to the live ChromaDB index (no request to do so; MiniLM's
   structured-extended model remains the deployed default).
6. **dynamic_alpha ablation run (`scripts/ablate_dynamic_alpha.py`,
   `results/dynamic_alpha_ablation.csv` + `..._bootstrap_significance.csv`)
   — an honest, mostly NULL result.** recall@1/recall@3/recall@5/MRR are
   byte-identical to the existing `retrieve_adaptive` (diff=0.0000 on every
   bootstrap resample) because the exact-match ceiling
   (`UNAMBIGUOUS_MATCH_SCORE`) already guarantees top-1 regardless of
   lambda_ for the unambiguous entity-heavy queries where it fires — lambda_
   only affects ordering below the ceiling. Only nDCG@10 showed a tiny,
   borderline-significant edge (diff=+0.0022, p=0.045 — right at the 0.05
   line). Root cause, also disclosed: `entity_signal_strength` never
   exceeded 0.25 on this test set (99 queries at 0.0, 101 at 0.25, none
   higher) — none of the 200 test queries name two-or-more independent
   entity signal types at once, so the "continuous" mechanism rarely gets
   to differ from a near-binary split in practice on THIS corpus's query
   distribution. This is a genuine, reportable finding (DAT's mechanism,
   adapted to avoid the raw-BM25-magnitude confound, still doesn't show a
   meaningful improvement here) — not a failure to hide, and not "rescued"
   by re-running with different lambda values without new evidence it
   would help.
7. **Confidence-ordered zig-zag context assembly ablated for the first time
   (`scripts/ablate_confidence_ordering.py` + `scripts/compute_metrics.py`
   + `scripts/confidence_ordering_significance.py`, n=98 open-ended
   non-abstained queries) — another honest NULL result.** Raw means
   slightly favored ordering OFF on all 4 metrics (BLEU 0.757 vs 0.726,
   ROUGE-L 0.856 vs 0.843, BERTScore 0.973 vs 0.971, METEOR 0.881 vs
   0.861), but paired significance testing (paired t-test + Wilcoxon) found
   NONE of these differences significant (all p > 0.2, most p > 0.35) —
   this is noise at n=98, not a real effect either direction. Most likely
   explanation: this project's assembled context is typically 2-5 chunks,
   far smaller than the 10-20+ document settings where Liu et al.'s
   Lost-in-the-Middle effect (and Jin et al. 2025's zig-zag fix for it)
   were originally measured — there may simply not be enough "middle" for
   the effect to bite at this scale. `use_confidence_ordering` stays on by
   default (matches deployed behavior, and this null result gives no
   reason to change it either way) but is no longer an untested claim.
8. **Ambiguous-entity notice quality, finally measured (not just the
   "Rahman" anecdote)** — `scripts/eval_ambiguous_entity_notice_quality.py`,
   n=55 genuinely-ambiguous queries x 3 conditions x LLM-judge, paired
   bootstrap significance (`results/ambiguous_notice_quality_summary.csv`,
   `..._raw.csv`). Real, nuanced result:
   - `no_notice` (pre-fix behavior): avoids_false_confidence=0.182,
     asks_for_clarification=**0.000** (never once asks — exactly the
     silently-confident-arbitrary-pick failure mode this whole feature
     exists to fix, now directly measured, not just described),
     offers_disambiguator=0.073.
   - `flat_notice` and `conditioning` BOTH significantly beat `no_notice`
     on avoids_false_confidence (+0.345 p<0.0005; +0.236 p=0.004) and
     asks_for_clarification (+0.565 p<0.0005; +0.528 p<0.0005) — the core
     "detect ambiguity, don't silently guess" fix is real and large.
   - `conditioning`'s own specific selling point — offers_disambiguator —
     nearly doubles over `flat_notice` (0.127->0.218) and IS significant
     vs. `no_notice` (p=0.012), but the conditioning-vs-flat_notice
     comparison itself does NOT reach significance at n=55 (diff=+0.093,
     p=0.088) — a promising trend, not yet a confirmed incremental win.
   - Honest, disclosed nuance: `conditioning` scored slightly LOWER than
     `flat_notice` on both avoids_false_confidence (0.418 vs 0.527) and
     asks_for_clarification (0.527 vs 0.564) — neither difference reaches
     clean significance (both CIs border zero), but it's a consistent
     direction worth noting: the extra conditioning-hint text may read as
     slightly more presumptive/specific, a small possible trade-off
     against its disambiguation benefit. Not swept under the rug.
9. **AmbigDocs-style answer-type re-scoring of the SAME 165 responses above**
   (`scripts/score_ambiguous_notice_ambigdocs_style.py` — no new
   generation, only a second LLM-judge pass, adapted from AmbigDocs/
   RAMDocs's Complete/Partial/Merged/No-answer ontology, both independently
   verified this session). Distribution (`results/ambiguous_notice_
   ambigdocs_style_summary.csv`):
   - `no_notice`: complete=0.000, partial=0.400, **merged=0.418**,
     no_answer=0.182 — the baseline's dominant failure mode under this
     stricter lens is CONFLATING different people's info into one false
     answer, not just failing to ask.
   - `flat_notice` and `conditioning`: identical AGGREGATE distributions
     (complete=0.018, partial=0.564, merged=0.236, no_answer=0.182) — both
     cut the merged-answer rate nearly in half vs. baseline (0.418->0.236).
     Verified this isn't a judge bug: per-query labels match on 53/55
     queries, and the other 2 flipped in OPPOSITE directions (AMB271
     partial->complete, AMB372 complete->partial) and happen to exactly
     cancel out in the aggregate count — a genuine coincidental tie at the
     marginal level, not identical underlying behavior.
   - **`complete` rate is near-zero in EVERY condition (0-1.8%).** Read
     this with its scoping caveat, not as a bare failure number: AmbigDocs'
     original task is multi-document QA where fully enumerating every
     valid interpretation IN ONE ANSWER is the goal; this system is
     deliberately designed to ask a SHORT clarifying question and let the
     conversation continue, not to preemptively dump every candidate's
     full details in turn one — so a near-zero "complete" rate under
     AmbigDocs' literal criterion may partly reflect a task-shape mismatch,
     not a straightforward system failure. Disclosed as exactly that if
     this number is ever used in the paper — do not present it as
     "the system fails at disambiguation 98% of the time" without this
     context.
   - `complete_rate` paired-bootstrap comparisons all show `significant=
     False` despite p=0.0000 for two of them -- a numerical boundary
     artifact (CI lower bound lands exactly at 0.000 because the base rate
     is so low), not a meaningful contradiction; treat the CI, not the raw
     p_approx, as authoritative per this project's own bootstrap-testing
     convention.
4. **New EXPERIMENTAL retrieval mode**: `HybridRetriever.entity_signal_
   strength()` / `retrieve_dynamic_alpha()` (`pipeline/hybrid_retriever.py`)
   — a continuous per-query fusion weight, inspired by DAT (Hsu & Tzeng
   2025, arXiv:2503.23013, verified via a background literature-research
   agent's direct WebFetch), replacing `retrieve_adaptive`'s binary
   route/lambda split. Adapted rather than ported as-is: DAT's own
   mechanism drives the continuous weight from raw BM25-vs-dense top-1
   score dominance, but a direct check on this corpus's own test set found
   that signal is INVERTED here (entity_heavy queries' raw BM25 top-1 mean
   = 17.8, LOWER than open_ended's 33.0 — a query-length confound, not a
   match-quality signal), so this uses the COUNT of independent structural
   entity-recognition signal types (course code / alias / faculty initial
   / faculty name) instead. Implemented, unit-tested for sane/monotonic
   behavior with zero GPU dependency (`tests/test_dynamic_alpha.py`, tests
   the method against a lightweight stand-in object rather than
   instantiating the real GPU-touching HybridRetriever). **NOT YET
   empirically validated** — `scripts/ablate_dynamic_alpha.py` is written
   and ready (reuses `measure_ir_metrics.py`'s exact metric/bootstrap
   methodology) but has not run yet (needs the GPU free from E5 training
   first). Does not change any existing caller's behavior.
5. **Literature agent findings worth acting on next** (all independently
   WebFetch-verified, see agent output for full citations): TRF (tensor-
   based re-ranking fusion, PVLDB 2025 arXiv:2508.01405) beats RRF by +8.1%
   nDCG@10 on a hybrid full-text+dense benchmark — a heavier lift than
   dynamic_alpha, not yet attempted. AmbigDocs (arXiv:2404.12447) and
   MADAM-RAG/RAMDocs (COLM 2025, arXiv:2504.13079) give a directly
   comparable answer-type ontology (Complete/Partial/No/Ambiguous/Merged)
   this project's own 55-query ambiguous-entity test set could be scored
   against instead of a homegrown metric — not yet implemented. Confirmed
   **GroundLM 2026 is a real EMNLP 2026 workshop** ("Grounding Language
   Models — Learning Faithfully and Efficiently"), not a fabricated venue
   name — live OpenReview page and CFP found.

## 2026-07-29 reviewer-response round (A/A*-readiness gaps)
User relayed a reviewer-style critique of what's blocking main-track
publication: (1) conformal abstention built but disabled, no calibration
data; (2) human annotation prepared but zero labels exist, gating both (1)
and quality claims generally; (3) single LLM-judge, no human validation;
(4) ambiguous-entity conditioning-vs-flat-notice trend not significant at
n=55. Explicit instruction: fix what's genuinely fixable, do not fabricate
what isn't. Response, each item scoped honestly:

1. **n=55 power issue — actually fixed, not just discussed.**
   `scripts/expand_ambiguous_entity_test.py` grew the ambiguous-entity test
   set from 55 to 220 real queries: same 55 genuinely-ambiguous people (real
   FacultyList name collisions), 4 distinct field questions each (office
   room / email / designation / status) instead of just one — real,
   distinct questions about the same real ambiguous people, not repeated
   padding. `scripts/eval_ambiguous_entity_notice_quality_expanded.py`
   reruns the EXACT same 3-condition/LLM-judge/paired-bootstrap methodology
   at this 4x larger n. Running as of this writing
   (`results/ambiguous_notice_quality_expanded_raw.csv` /
   `..._summary.csv` once complete) — result not yet known, report
   whatever it actually shows.

2. **Conformal calibration — built a legitimate non-human, non-self-judged
   path, explicitly NOT a substitute for the real human-labeling task.**
   `scripts/build_conformal_calibration_labels.py` derives (claim, context,
   is_correct) labels from two STRUCTURAL signals, neither requiring a
   human or any model's semantic self-judgment: negative labels from
   Category="Out of Scope/Unanswerable" queries (any confident claim here
   is wrong by the dataset's own construction, since no answer exists);
   positive labels from queries where `question_match_any` verifies the
   exact source record was retrieved, cross-checked against the known
   reference_answer via plain deterministic word overlap (no NLI, no LLM
   judgment — avoids the exact circularity being criticized).
   `scripts/run_conformal_calibration.py` (written, will run once labels
   are collected) feeds this into the already-existing
   `calibrate_threshold_from_labels()` and reports the result honestly,
   including if the label set turns out too small/imbalanced to trust.
   **Explicitly disclosed, in the script's own output**: this does NOT
   cover `results/human_annotation_sample.csv`'s 50 disagreement
   -prioritized cases (still unlabeled, still needs a person) — those were
   selected specifically because they're hard cases a structural heuristic
   can't resolve. This closes part of the gap, not all of it.

3. **Single-LLM-judge exposure — partial mitigation, honestly scoped as
   partial.** `scripts/judge_reliability_check.py` (written, queued to run)
   re-scores the SAME already-generated ambiguous-entity responses with a
   deliberately reworded (not just re-asked) judge prompt — same three
   criteria, different phrasing/order — and reports percent agreement +
   Cohen's kappa between the original and reworded verdicts. This tests
   prompt-rewording robustness (a real, checkable property), NOT agreement
   with ground truth or a human — a high kappa would mean the judge is
   self-consistent, not that it's correct. Disclosed as such in the
   script's own printed output, not just here.

All three scripts are additive/new — nothing about the existing deployed
system changed as a result of writing them; `use_conformal_backoff` stays
False until (2) actually produces a trustworthy threshold.

## 2026-07-30/31 continuation: judge reliability check — a serious finding
`scripts/judge_reliability_check.py` re-judged all 660 already-generated
ambiguous-entity responses (results/ambiguous_notice_quality_expanded_raw.csv)
with a reworded, reordered (but semantically identical) judge prompt, to
test whether the LLM-judge's verdicts are a stable property of the response
or an artifact of exact prompt phrasing. Result
(`results/judge_reliability_summary_expanded.csv`):
- `avoids_false_confidence`: percent_agreement=0.441, Cohen's kappa=**0.114**
  — barely above chance (Landis & Koch: <0.2 is "slight" agreement). The
  verdict flips on prompt rewording more than half the time. **Every
  avoids_false_confidence number reported anywhere in this project
  (original n=55 eval, expanded n=220 eval) should be treated as measured
  with an unreliable instrument** — do not present those specific numbers
  as trustworthy without this caveat attached.
- `asks_for_clarification`: agreement=0.992, kappa=0.984 — near-perfect,
  reliable.
- `offers_disambiguator`: agreement=0.868, kappa=0.656 — substantial,
  reasonably reliable.

This does NOT invalidate the core "the fix helps" finding (asks_for_
clarification alone, rock-solid at kappa=0.984, already carries that
result), but it does mean the specific avoids_false_confidence deltas
(e.g. flat_notice vs no_notice +0.196 at n=220) should not be quoted as
precise or even directionally trustworthy without this reliability caveat
-- a genuinely different and more serious problem than "no human validated
the judge," since this is a MEASURED failure, not just an unaddressed risk.

## Conformal calibration, second attempt: adversarial context-swap
After the first calibration attempt failed (Out-of-Scope-query mining found
the system mostly refuses/redirects correctly rather than hallucinating,
so too few genuine negative examples existed to calibrate against --
see below), built `scripts/build_conformal_adversarial_negatives.py`:
deliberately feeds an open-ended query its own real answer's WRONG
same-category "confusor" context (a different real Q/A pair, same
Category, different Answer) and checks whether the generated response
states the confusor's specific facts as if they answered the real
question -- a structurally verifiable hallucination (ground truth is
which Answer text the claim overlaps with), not a judgment call.

Result: only 1 negative example survived out of 80 same-category confusor
pairs (`results/conformal_adversarial_negative_labels.csv`), and even that
single survivor is debatable on inspection -- the flagged claim ("the
library provides journals/magazines for in-library use") is plausibly just
a true, reasonable restatement, not a clean hallucination. **CONCLUSION:
two independent automatic negative-mining strategies (organic Out-of-Scope
mining, adversarial same-category context-swap) were tried and BOTH failed
to produce a usable negative-example set** -- not from lack of effort, but
because this system's generation behavior (correct refusals on genuinely
unanswerable questions; graceful non-commitment on mismatched context) is
good enough that it doesn't reliably produce the confident-wrong-answer
failure mode conformal calibration needs examples of. Conformal abstention
CANNOT be calibrated by automatic means with what's been tried -- it
genuinely needs real human-labeled data (results/human_annotation_sample.
csv, still blank) or a stronger/different perturbation strategy not yet
designed. This is now a settled, well-evidenced conclusion, not an open
question to keep re-attempting without a new idea.

## Infrastructure note: Ollama model-path environment bug (found + fixed 2026-07-31)
Two background jobs failed instantly with `404 Client Error: Not Found for
url: http://localhost:11434/api/generate` / `"model 'llama3.1:8b' not
found"`. Root cause: Ollama's model data has always lived at
`E:\ollama_models` (a persistent, User-level `OLLAMA_MODELS` env var), but
the actual running `ollama.exe serve` process (a child of a tray-app
supervisor, `ollama app.exe`, itself a Windows startup item) had been
launched at some earlier point before/without that env var in its own
process environment, so it was silently serving from the empty default
`C:\Users\<user>\.ollama\models` instead -- `ollama list`/`/api/tags` both
showed zero models even though nothing was actually lost (the real model
files, including the 4.9GB blob, were untouched at `E:\ollama_models`).
Fixed by killing the whole supervisor chain (`ollama app.exe` + its
`ollama.exe serve` child) and relaunching `ollama serve` from a shell
session confirmed to have the correct `OLLAMA_MODELS` in its own process
environment -- verified with a real end-to-end generate call afterward,
not just the /api/tags listing. If Ollama calls ever fail with a 404
"model not found" again despite the model clearly existing on disk, check
this first before assuming a code bug: `Invoke-RestMethod http://localhost:
11434/api/tags` showing an empty model list, with the model files
confirmed present under the real `OLLAMA_MODELS` path, is the signature.

**Recurred the same day** after a session/process interruption (the
conditioning-hint-v2 eval died silently mid-run when the harness session
was interrupted; on relaunch, `/api/tags` again showed `{"models":[]}`).
Confirmed via `[System.Environment]::GetEnvironmentVariable("OLLAMA_MODELS",
"User")` that the var IS correctly persisted at the Windows User level --
so the root cause isn't a missing/lost env var, it's that `ollama app.exe`
(a Windows startup item) doesn't reliably pick up the persisted value on
every relaunch, e.g. if it's restarted by Windows session/Explorer
machinery without a full logoff that would refresh its inherited
environment block. Same fix applied (kill via `Stop-Process`, found via
PowerShell `Get-Process` since `ps aux` in this Git-Bash environment
does NOT reliably see Windows GUI/tray processes -- relaunch `ollama.exe
serve` from a shell with the correct env, verify with a real generate
call). Given it has now recurred once already, treat this as a
standing, re-checkable risk after ANY session interruption or machine
restart, not a one-time incident -- check `/api/tags` before trusting
any Ollama-dependent job's early failures.

**Separately found the same day**: Git-Bash's `ps aux` in this
environment gave inconsistent/wrong PIDs for backgrounded Windows python
processes across repeated calls (three different monitor liveness checks
built around `ps aux`/`ps -p` all false-flagged a genuinely-alive,
`Get-Process`-confirmed process as dead). `tasklist` from Bash was also
tried but failed inside a Monitor-tool subshell specifically (likely a
PATH difference in that subshell). The reliable fix was to stop relying
on process-liveness checks from Bash/Monitor entirely and instead gate
on LOG CONTENT only (a completion marker the script itself prints, or a
`Traceback`) -- if genuine liveness confirmation is ever needed, use
PowerShell `Get-Process -Id <pid>` from the main shell, not a Bash-side
check.

## 2026-07-31: Banglish dataset expansion + corpus-specific stopword fix
User provided a new Banglish QA dataset (2050 rows, 160 real bracu.ac.bd
pages, genuine collected FAQ content, not synthetic). Ingested via
`scripts/ingest_new_banglish_dataset.py`: deduplicated (59 exact dupes
dropped), zero overlap with existing BanglishQA confirmed directly, 1991
genuinely new rows added with URL-derived categories (42% fell into an
honest "General / Web FAQ" fallback rather than being force-fit) and a
fixed-seed stratified train/val/test split. BanglishQA: 1053 -> 3044 rows.
Corpus rebuilt (5059 -> 7050 chunks), BM25 + Chroma indexes rebuilt,
verified end-to-end with a real retrieval query before declaring done.

**Real bug + real improvement found via corpus-specific TF-IDF stopword
analysis** (`scripts/derive_corpus_specific_banglish_stopwords.py`,
motivated by a verified literature finding that corpus-derived stopword
lists beat generic ones for code-mixed IR): "kore" (a common Bengali verb
form, 13.6% of questions) was never collapsing into the existing "kor"
normalization group alongside every OTHER inflection of the same verb --
a real gap, fixed. Also added 15 genuine function words to
`BANGLISH_STOPWORDS` (kor, ache, hobe, hoy, jay, koto, kothay, kivabe,
der, te, ke, ami, kon, pabo, pare) after manually confirming each is a
grammatical/auxiliary word, not domain content -- candidates that turned
out to be real content words (brac, student, course, semester, library,
...) were deliberately excluded despite also crossing the frequency
threshold. BM25 index rebuilt; verified via direct tokenization test and
the existing regression suite (`tests/test_patterns.py`, `tests/test_
dynamic_alpha.py`, both still passing).

Five parallel jobs were launched to validate/improve further (embedding
retrain, the avoids_false_confidence judge fix applied to the full
660-row set, a third conformal-calibration attempt, zig-zag at double
context size, an improved conditioning-hint phrasing). The embedding
retrain finished cleanly and is documented above. **The other four,
running concurrently against the same local Ollama instance, exposed a
real, serious bug — documented in the next section — and were killed
before completing.**

## 2026-07-31: judge-parsing silent-corruption bug found and fixed
Running 4 Ollama-dependent scripts concurrently (this project's own
overnight parallel-ablation pattern) caused a genuine request-queueing
pileup severe enough to produce multiple outright timeouts/crashes even
after `post_with_retry`'s exponential-backoff retry was added. That fix
only catches HTTP-level failures (`ReadTimeout`/`ConnectionError`) — it
does **not** catch a request that returns 200 OK with a malformed,
truncated, or otherwise unparseable body, which is exactly what
contention can produce without raising any exception at all. Every
judge/extraction function in this project that parsed such a response
was, until today, silently defaulting to a **specific, fabricated**
score (e.g. `n_names=0`, `avoids_false_confidence=1`, label=
`"ambiguous_unclear"`) whenever its expected field markers weren't found
— indistinguishable, after the fact, from a genuine judgment. This was
flagged by direct user review, not discovered internally, and is a real
data-corruption risk, not a hypothetical one: it meant **the stored
avoids_false_confidence judge validation (kappa=0.7548, `results/
improved_afc_judge_check.csv`) had to be treated as unverified** — 28/150
rows show the `n_names=0 AND acknowledges=0` pattern consistent with
either a genuine judgment or a silent-default artifact, and raw judge
response text wasn't logged at the time so this can't be resolved
retroactively.

**Fix**: added `pipeline.ollama_client.JudgeParseError` (an exception
carrying the raw unparseable response text) and made every
judge/extraction function raise it instead of defaulting, across all
five files that judge/extract from Ollama responses:
`improve_avoids_false_confidence_judge.py`,
`apply_fixed_avoids_false_confidence_judge.py`,
`judge_reliability_check.py`, `eval_ambiguous_entity_notice_quality.py`,
`score_ambiguous_notice_ambigdocs_style.py` (this last one needed care:
its `"ambiguous_unclear"` is a legitimate ontology label, not a parse
-failure marker — the fix only raises when the response matches NONE of
the 5 possible labels at all, since a genuine `ambiguous_unclear`
verdict is still caught by the normal label-matching loop). Each
caller now retries the parse up to 3 times (a malformed response is
plausibly a one-off fluke under contention, not necessarily systematic),
and rows that still fail to parse are recorded with an explicit
`parse_failed=True` flag and **excluded** from kappa/summary/
significance calculations — never silently defaulted, never silently
dropped without a count. Paired-bootstrap significance code that pivots
per query_id x condition was also fixed to `dropna` on incomplete
query_ids after filtering, since a parse failure hitting only one
condition would otherwise misalign the paired arrays.

**Remediation applied, in the order the user specified**: (1) killed
all 4 concurrent Ollama-dependent jobs, confirmed via process list that
nothing remained running; (2) confirmed via `/api/tags` that Ollama was
alive and the model registered, and via `/api/ps` that nothing was
actually loaded in memory (`{"models":[]}`) — validating the cold-start
-cost concern directly rather than assuming it; (3) implemented the
parse-failure-detecting fix above; (4) re-ran the n=150 validation as a
single clean job (not concurrent with anything else).

**Verified result (2026-07-31, single clean job, `results/improved_afc_
judge_check.csv` / `improved_afc_judge_summary.csv`)**: kappa=0.7471,
raw agreement=0.8921, on n=139 (11/150 = 7.3% of rows genuinely failed
to parse even after 3 retries each, and were honestly excluded, not
defaulted). This **replaces** the old, uncertain kappa=0.7548 figure as
the number to cite — it is close to the old figure but is now actually
trustworthy, since this run logged raw response text and used a parser
that raises rather than defaults on failure. The ~7.3% genuine
parse-failure rate is itself worth disclosing alongside the kappa in
any writeup: even under clean single-job conditions with no contention,
Llama-3.1-8B does not reliably produce the requested extraction format
on every call.

**Full 660-row re-score, re-run clean (2026-07-31, single job, `results/
ambiguous_notice_quality_expanded_raw_fixed_afc.csv` / `..._summary_
fixed_afc.csv` / `..._significance_fixed_afc.csv`)**: 0/660 rows failed
to parse -- notably ZERO, vs. 11/150 (7.3%) on the validation run above.
This is real evidence the parse-failure rate itself is mostly a
CONTENTION artifact (queueing under concurrent load producing truncated
completions), not a property of the prompt/model alone -- the validation
run above and this re-score used the same code, same model, same
prompt; the only material difference was this re-score ran strictly
alone. Verdict changed on 35.2% of the 660 responses vs. the old,
corrupted-risk holistic judge -- a large shift, underscoring why this
needed a clean redo rather than being trusted as-is.

Real, now-trustworthy numbers (n=220 query_ids complete across all 3
conditions): avoids_false_confidence mean = no_notice 0.436, flat_notice
0.809, conditioning 0.800. flat_notice vs no_notice: diff=+0.373,
p<0.001, significant. conditioning vs no_notice: diff=+0.364, p<0.001,
significant. **conditioning vs flat_notice: diff=-0.009, not
significant.** This is the THIRD criterion (after asks_for_clarification
and offers_disambiguator, both already null) to show no measurable
benefit from the conditioning-attribute hint extension over the simpler
flat notice -- a consistent, replicated null result across all three
judged criteria now, not one criterion's fluke. The base ambiguous
-entity fix (notice vs. no notice at all) remains a large, robust,
now-doubly-verified win; the conditioning extension specifically does
not currently earn its complexity and should be reported as a tested,
disclosed null rather than quietly dropped or force-fit into a positive
result.

## Known still-open items (real, not resolved by more engineering alone)
1. **Human hallucination annotation** — still needs a person. A
   disagreement-prioritized 50-item sample is prepared
   (`results/human_annotation_sample.csv`, blank rating columns). Also now
   the blocking dependency for calibrating `conformal_abstention.py`'s
   threshold (see above) — one human-labeling pass would unblock both.
2. **Adaptive routing's own contribution adds no measurable benefit** over
   a simpler fixed-lambda + exact-match configuration (replicated across
   THREE independently-trained embedding models now, including this
   round's structured-extended one) — genuine negative result, don't
   "rescue" without new evidence. **Re-verified 2026-07-31 with fresh data**
   under the CURRENT (Banglish-expanded) embedding model and the current
   7050-chunk corpus (the prior evidence file was dated 2026-07-28, from
   before that model existed — `scripts/measure_ir_metrics.py` +
   `scripts/isolate_adaptive_routing.py` re-run clean; the previous copy
   of `results/ir_metrics.csv` was stale by the same margin as the
   generation-quality table in item 4 below). Entity-heavy-only isolation
   (n=100, the only subset where adaptive's routing decision can possibly
   differ from a fixed baseline by construction): adaptive ties
   `bm25_only` and `full_hybrid` EXACTLY on recall@1/3/5/MRR (100/100
   identical predictions both comparisons); nDCG differences are mostly
   non-significant, with one metric (`ndcg@10` vs `bm25_only`) reaching
   p=0.036 but in adaptive's DISFAVOR (diff=-0.0062, tiny). The null holds
   up under the improved embedding model, not just the older one — this
   strengthens rather than weakens the finding's credibility. `results/
   adaptive_routing_isolated_significance.csv` now reflects this current
   run, not the 2026-07-28 one.
3. **RESOLVED/UPDATED 2026-07-31**: the finer lambda sweep (11 points,
   0.0-1.0, 60-query stratified subset) was stale (Jul 26, pre-Banglish
   -expanded model) -- archived that raw data (`archive/lambda_sweep_raw_
   outputs_pre_banglish.csv`) and regenerated it fresh under the current
   model (`scripts/run_lambda_sweep.py`, made checkpointed/resumable and
   flush-fixed first, same infra lessons as run_ablation.py the same day;
   660/660 rows, `results/lambda_sweep_raw_outputs.csv`).

   Real, current, significant confirmation of a genuine mid-blend peak
   (NOT flat/monotonic between the two single-method endpoints): on
   open-ended queries, **λ=0.3 significantly beats bm25_only on ALL four
   metrics** (bleu diff=+0.105 p=0.008, rougeL diff=+0.076 p=0.010,
   bertscore diff=+0.010 p=0.008, meteor diff=+0.056 p=0.013, n=30), while
   λ=0.7-0.9 are significantly WORSE than vector_only on all four metrics
   (p=0.03 each) -- a real peak around λ=0.2-0.3, not noise, replicating
   and updating the earlier λ≈0.25 finding with fresh current-model
   evidence (`results/lambda_sweep_significance.csv`, 32/216 comparisons
   significant). The deployed λ=0.5 is still not the empirically best
   fixed weight for open-ended queries on the TUNING set -- but see the
   held-out check immediately below, which does NOT confirm this as a
   real, generalizable effect.

   **Held-out validation (2026-07-31, `scripts/eval_lambda_held_out.py`,
   `results/lambda_0.3_held_out_significance.csv`): λ=0.3 does NOT
   replicate on genuinely unseen data.** The 60-query tuning subset above
   was used ONLY to select λ=0.3; this check evaluated it on the
   remaining 140 queries the selection process never touched (reusing
   the existing full_hybrid/bm25_only/vector_only/no_retrieval
   generations for those same 140 queries from the main ablation table,
   generating only the new λ=0.3 arm). Result: on open-ended queries
   (where the tuning set found a significant win), the held-out set
   shows NO significant difference between λ=0.3 and bm25_only/
   full_hybrid on any metric. On entity-heavy queries, λ=0.3 is
   significantly WORSE than both bm25_only (bleu p=0.008, bertscore
   p=0.021, meteor p<0.001) and full_hybrid (bleu p=0.010, meteor
   p=0.002). **Conclusion: the tuning-set peak was very likely sample
   -specific noise, not a real, generalizable effect. Do NOT change the
   deployed λ=0.5 based on the tuning-set finding alone -- this is
   precisely the overfitting risk the "needs a held-out pass" caveat
   above was warning about, and the held-out check just caught it.**
   This is now a closed, doubly-tested question: the tuning-set finding
   is real (it happened, on that data), but it does not generalize, and
   the deployed fixed weight should not be changed on its basis.
4. **RESOLVED 2026-07-31**: the full 200-query, 4-config generation-quality
   ablation table has now been regenerated under the current (Banglish
   -expanded) embedding model -- `scripts/run_ablation.py`, made
   checkpointed/resumable first (see infra notes above), 800/800 rows,
   `results/ablation_raw_outputs.csv`. Metrics + significance recomputed
   into the canonical files (`ablation_metrics_summary.csv`,
   `ablation_metrics_per_query.csv`, `significance_tests.csv`), replacing
   the stale versions.

   Real result: full_hybrid=(bleu 0.6624, rougeL 0.8293, bertscore 0.9648,
   meteor 0.8515), bm25_only=(0.6655, 0.8318, 0.9664, 0.8631),
   vector_only=(0.6409, 0.8226, 0.9648, 0.8322), no_retrieval=(0.0460,
   0.1907, 0.8648, 0.2898). Sanity check passes cleanly: full_hybrid vs
   no_retrieval is hugely significant on every metric, both overall and
   per-subset (p<0.0001). full_hybrid vs bm25_only: NOT significant
   anywhere overall (all p>0.15) -- consistent with, and now doubly
   corroborated by, the same-day adaptive-routing re-verification above
   (item 2) that found the two ANSWER much the same on this corpus.
   full_hybrid vs vector_only: not significant overall, but IS
   significant on the entity-heavy subset specifically (bleu diff=+0.069,
   p=0.011; meteor diff=+0.056, p=0.0024) -- the lexical/exact-match
   signal full_hybrid adds over pure vector search earns its keep exactly
   where expected (entity lookups), not on open-ended queries (all n.s.
   there). This is now the real, current, citable table -- not a stale
   one from an earlier embedding model.
5. **RRF's `k` constant and adaptive routing's design still cannot be
   tested under genuinely ambiguous ENTITY-CODE resolution** (course codes
   shared across cross-listed sections) — the FACULTY-NAME version of this
   gap (item 6, previous version of this file) is now closed by this
   round's ambiguous-entity work; the course-code version is not, since
   this corpus has no cross-listed-code collisions to test against.
6. **Governance-category augmentation and alias-table population both
   need the user's decision/input** — the former has a ready-to-review
   proposed file, the latter needs genuinely new source data this repo
   doesn't have.

## 2026-07-31: zig-zag confidence-ordering re-tested at double context size -- still null
`scripts/ablate_confidence_ordering_larger_context.py` re-tested the zig-zag
"sandwich" context ordering (pipeline/novel_pipeline.py's
use_confidence_ordering) at final_k=10 (double the deployed default of 5),
to test the specific hypothesis that the original null result (n=98,
final_k=5) was an artifact of contexts being too small for the Lost-in-the
-Middle position effect to matter. Run cleanly as a single job (98 eligible
open-ended, non-abstained queries x 2 conditions = 196 rows, mean 10.85
context pieces per answer, confirming the larger context was actually
exercised, not just requested). Scored with the same BLEU/ROUGE-L/
BERTScore/METEOR + paired significance method as every other generation
-quality ablation in this project (`results/confidence_ordering_larger_
context_metrics_per_query.csv` / `..._summary.csv` /
`..._significance.csv`).

**Result: still null.** ordering_on numerically higher on all 4 metrics
(bleu +0.015, rougeL +0.009, bertscore +0.003, meteor +0.007) but NONE
significant (paired t-test p=0.40-0.70, Wilcoxon p=0.70-0.96, n=98). This
is a STRONGER, more specific negative finding than the original: it rules
out "context too short for the mechanism to matter" as the explanation,
one of the two hypotheses this project's own docstrings proposed for the
first null result. The most likely remaining explanation (not yet tested,
and possibly untestable without a differently-designed corpus) is that
this system's context chunks are already individually short and
information-dense (structured field:value rows or short QA-pair
paragraphs, not long free-text passages), so there may not be a
meaningful "middle" for position bias to degrade in the first place --
zig-zag ordering is a real, correctly-implemented mechanism (see
tests/test_conformal_abstention.py's dedicated regression test) that
simply doesn't have a problem to solve in this specific corpus's context
shape. Report as a genuine, twice-tested null, not a bug or an
underpowered check.

## 2026-07-31: conditioning_v2 (improved hint phrasing) -- a real, verified win
Follow-up to the conditioning-vs-flat-notice null (above): tested an
IMPROVED, more directive conditioning-hint phrasing (`build_conditioning_
hint_v2`, `scripts/eval_conditioning_hint_v2.py`) that gives the model a
concrete question template using the actual distinguishing values (e.g.
"ask: Which one do you mean: Professor or Lecturer?") instead of the
original's conditional/passive instruction to check the user's own
phrasing. Same n=220 expanded ambiguous-entity set, 4 conditions
(no_notice/flat_notice/conditioning/conditioning_v2), checkpointed +
resumable script (added after two real infra failures mid-run -- see
below), run to completion: 880/880 rows, 0 parse failures.

**A real methodological catch before trusting the result**: the initial
significance pass used `judge_response_or_fail` imported from `eval_
ambiguous_entity_notice_quality.py` -- the ORIGINAL holistic avoids_
false_confidence judge (kappa=0.114, already known near-chance-reliable),
not the validated decomposed judge (kappa=0.7471) built and applied to
the main 660-row dataset earlier the same day. This script simply
predated that fix and was never updated to use it. The first pass showed
a striking, counterintuitive result: conditioning_v2 significantly WORSE
than flat_notice on avoids_false_confidence (diff=-0.227, p<0.0001) while
significantly BETTER on the other two criteria -- a pattern that looked
like a real trade-off but smelled like the known judge-reliability bug
given the criterion involved. Manually inspecting responses scored
avoids_false_confidence=0 confirmed it: replies like "Which one do you
mean: C. Lecturer (Contractual) or Lecturer (Full Time)?" -- a textbook
correct clarifying question -- were being scored as NOT avoiding false
confidence, backwards per the criterion's own definition.

Re-scored avoids_false_confidence for all 880 rows with the validated
decomposed judge (`scripts/apply_fixed_afc_judge_conditioning_v2.py`,
same extraction method as the main dataset fix, single clean job, 0
parse failures). **Verdict changed on 43.3% of responses** -- confirming
the original judge really was that unreliable here, not just on the
earlier dataset. The corrected result reverses the apparent regression
entirely:

Real, final, verified numbers (n=220 query_ids complete across all 4
conditions; `results/conditioning_hint_v2_summary_fixed_afc.csv` /
`..._significance_fixed_afc.csv` for avoids_false_confidence,
`results/conditioning_hint_v2_summary.csv` for the other two criteria
which were already reliable and unaffected):
- avoids_false_confidence: no_notice=0.486, flat_notice=0.805,
  conditioning=0.818, conditioning_v2=0.864. conditioning_v2 vs
  flat_notice: diff=+0.059, p<0.0001, **significant**. conditioning_v2
  vs conditioning: diff=+0.045, p<0.0001, **significant**.
- asks_for_clarification: conditioning_v2 vs flat_notice diff=+0.086,
  p<0.0001, significant; vs conditioning diff=+0.100, p<0.0001,
  significant.
- offers_disambiguator: conditioning_v2 vs flat_notice diff=+0.182,
  p<0.0001, significant; vs conditioning diff=+0.176, p<0.0001,
  significant.

**conditioning_v2 significantly outperforms both flat_notice and the
original conditioning hint on all three judged criteria.** This is a
genuine, hard-won positive result: the original conditioning-attribute
idea was correctly reported as null, a redesigned, more directive
phrasing was tested as a real follow-up hypothesis (not a re-run of the
same thing hoping for a different number), and the positive result that
emerged was independently re-verified against a judge-reliability
concern before being trusted -- it survived that check. Recommend citing
`conditioning_v2`'s phrasing (not the original `conditioning`) as this
project's conditioning-attribute mechanism in any writeup.

**Two real infra bugs found and fixed during this run** (both now fixed
project-wide, not just here): (1) `post_with_retry` didn't retry
transient HTTP 5xx errors (only connection-level exceptions) -- a real
500 from Ollama under load crashed a clean run outright; fixed to retry
500/502/503/504 specifically (not a blanket 5xx, since 501/505 aren't
transient). (2) The Windows machine's C: drive had only 1.2GB free,
which was capping the system pagefile/virtual-memory commit limit and
caused Ollama's OWN model load to fail with `CUDA_Host buffer
allocation` errors, independent of GPU/VRAM -- confirmed by the exact
same failure persisting even with `CUDA_VISIBLE_DEVICES=""` forcing
CPU-only mode. Fixed by relocating/expanding the pagefile to a drive
with free space and rebooting; verified after via real generation calls,
not just `/api/tags`. `scripts/eval_conditioning_hint_v2.py` was also
made checkpointed/resumable (append-as-you-go CSV writes instead of
one write at the end) specifically because of these failures --
worth keeping as the pattern for any future long Ollama-dependent batch
job on this machine.

## Infrastructure note: BERTScore (roberta-large) segfaults on GPU when Ollama is also resident
`scripts/compute_metrics.py` reproducibly segfaulted (exit 139, twice in a
row, same command) loading `roberta-large` on CUDA while Ollama's 8B model
was resident (2.3GB used, only ~1.6GB free on this 4GB card) -- likely a
native OOM inside a C++ extension that segfaults the whole process instead
of raising a catchable Python exception, rather than a bug in the script
itself. Fix: temporarily unload Ollama's model first (`curl .../api/generate
-d '{"model":"llama3.1:8b","prompt":"","keep_alive":0}'` -- frees the GPU
immediately, `nvidia-smi` confirmed 0MiB used afterward), then re-run;
succeeded immediately with ~4GB free. Ollama reloads automatically and
transparently on its next real request (confirmed: no code or config
change needed). Keep this in mind for any other GPU-heavy one-off script
(BERTScore, a reranker, etc.) run while Ollama is loaded on this machine.

## Output discipline (unchanged)
- Every experiment gets its own script and its own output file (CSV/JSON)
  saved under `results/` — don't just print to console and lose it.
- When reporting a result back for the paper, always point to the specific
  file and row/line it came from.
- `results/v1_naive_bm25/` holds a deliberately-preserved prior-baseline
  snapshot — keep treating any similarly superseded result set as a
  labeled historical baseline, not something to silently overwrite.
- Before trusting ANY externally-supplied citation or claim (pasted from
  outside this session), verify it independently first — this round found
  real fabrications in exactly this scenario. Don't skip this because a
  document looks well-formatted or confident.
