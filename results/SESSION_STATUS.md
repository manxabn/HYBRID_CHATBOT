# Session status — paused 2026-07-28 (user-requested pause, "restarting you soon")

User asked me to save progress and wait. This is NOT the end of the session — read this file
first on resume and continue exactly where this leaves off. This supersedes everything below
the "=== OLDER, SUPERSEDED CONTEXT ===" marker at the bottom of this file (kept for history only).

## Why this round of work started

User asked "what are our achievements and weaknesses" — I gave an honest list (see conversation).
User's response: **"remove all the weaknesses... without losing anything... read papers and fix
them... take your time."** Also, separately: **"I have gpu, use it."** This file tracks progress
against that specific weakness list.

## Big-picture wins this segment (all verified, nothing fabricated)

1. **GPU is now live.** RTX 3050 Ti (4GB VRAM), CUDA 12.1 torch installed into an isolated venv at
   `.venv/` (entirely on E:, never touched C:'s tight ~4GB free). `scripts/env_setup.ps1` now
   prepends `.venv/Scripts` to PATH, so plain `python` after sourcing it uses the GPU build.
   **Always run `. .\scripts\env_setup.ps1` first in every new PowerShell call.**

2. **LaTeX toolchain installed** (MiKTeX via winget, ~0.5GB on C:, acceptable). `paper/paper.tex`
   compiles cleanly to a 15-page PDF. `paper/SCITEPRESS.sty` is a **local structural stand-in**
   I wrote (NOT the real conference class file) so compilation works — replace with the real one
   before actual submission. pdflatex path:
   `C:\Users\manxa\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`. To recompile:
   `cd paper; & "<path>\pdflatex.exe" -interaction=nonstopmode -halt-on-error paper.tex` (run
   twice more after edits to resolve cross-refs/citations, no bibtex needed — bibliography is a
   manual `thebibliography` block, not a `.bib` file).

3. **Cross-encoder reranker fine-tuned** on this corpus's own hard negatives (real retrieved
   near-misses from `HybridRetriever.retrieve_adaptive`, not random text) via
   `scripts/finetune_reranker.py` → `models/finetuned_reranker_domain/` (loss 1.226→0.0055).
   `pipeline/reranker.py` auto-detects and uses it now (same pattern as the embedding model).
   **Not yet re-ablated against full_hybrid to confirm it flips from negative to positive** —
   this is queued in the Round L work below.

4. **MAJOR retrieval bug fixed: exact-match coverage was badly incomplete.** Investigated why
   only 40/100 English entity-heavy test queries had exactly one exact-match candidate (the
   condition needed for the `UNAMBIGUOUS_MATCH_SCORE` guarantee to fire). Found two root causes:
   - **20 queries had ZERO matches**: all were faculty lookups (email/room/designation) by name
     or initial — `_exact_match_ids` only ever recognized course-code patterns, never faculty
     identity, even though the `FacultyList` table (223 records — **a 7th corpus table I'd
     missed in every earlier count, need to fix paper.tex's dataset section**) was already
     ingested into the corpus and just never reachable via exact match.
   - **40 queries had 2-5 tied matches**: all were "prerequisites for X" / "coordinator for X"
     shaped queries, where the old code applied a blanket CourseDetails-cap-3 rule regardless of
     which table the query actually wanted, instead of routing "prerequisite"-keyword queries to
     the Prerequisites table and "coordinator"-keyword queries to the Coordinator table.

   **Fix applied** in `pipeline/hybrid_retriever.py`: added `faculty_initial_index` +
   `faculty_name_index` (built from FacultyList at init) and `PREREQ_KEYWORD_RE`/
   `COORDINATOR_KEYWORD_RE` table disambiguation in `_exact_match_ids`; also extended
   `is_entity_heavy()` to recognize faculty mentions for routing. **Result: 0 zero-match, 0
   multi-match out of 100 (was 20 and 40)**. Isolated ablation
   (`scripts/test_unambiguous_match.py`, rerun with corrected `_is_correct` that also checks
   email tokens and faculty initial/name identity, not just course codes):
   **top-1 accuracy 0.500→0.850** with the ceiling on (delta from the ceiling itself: was
   +0.08, now +0.34). Written to `results/unambiguous_match_test.csv`.

5. **FacultyAvailability test queries built** (`data/test_queries_faculty.csv`, 50 queries: 30
   room + 20 day-schedule, seed=42) — closes the previously-disclosed "never tested" gap. **Not
   yet run through the full pipeline for scored metrics** — queued below.

6. **Cross-lingual stress test built AND evaluated — turns a disclosed weakness into a genuine
   positive result.** The earlier null result for query translation (main Banglish test set) was
   explained as a test-set artifact (test queries are sampled from the corpus's own stored
   questions, so there's no real cross-lingual gap to bridge). Built a proper stress test
   (`scripts/build_crosslingual_stress_test.py` → `data/test_queries_crosslingual_stress.csv`, 9
   queries: EnglishQA test-split rows whose answer has NO match anywhere in BanglishQA, verified
   by DB query, deduplicated by unique answer, then LLM-rephrased into Banglish — construction
   method fully disclosed, not presented as organic data). Evaluated
   (`scripts/eval_crosslingual_stress.py`, retrieval-only, top-5 accuracy via reference-answer
   substring match): **plain retrieval 33.3% (3/9) vs. with query translation 77.8% (7/9)**.
   n=9 is small — report as suggestive/qualitative, NOT significance-tested — but this is a
   real, honest, mechanistically-explained validation that query translation genuinely works in
   the scenario it was designed for. Written to `results/crosslingual_stress_eval.csv`.

7. **Abstention recalibrated — both disclosed weaknesses substantially improved with real data.**
   Reran `scripts/calibrate_abstention.py` (retrieval-only, cheap) after the retriever fix above,
   AND added `build_synthetic_entity_calibration_set()` to it: template-generated but
   **DB-verified** additional labeled examples for the entity_heavy route specifically —
   nonexistent course codes / faculty initials (verified absent from the DB = genuinely
   should-abstain) paired with real ones that do have data (genuinely answerable). This is
   disclosed in the script's docstring and must be disclosed in paper.tex too (these are
   template-generated, not organic student questions).
   - **entity_heavy: n=18→60, accuracy 0.889→0.950, 95% CI (0.65,0.99)→(0.861,0.990)** (much
     tighter).
   - **open_ended: n=362→372, recall 0.457→0.870, accuracy 0.666→0.710** (this alone resolves
     the "misses more than half of true out-of-scope questions" weakness).
   New thresholds written to `results/abstention_threshold.json` (already the live file
   `AbstentionGate` loads — **no code change needed, already active**).

## What's currently RUNNING when this pauses

**Background task `b1a243mbl`**: `python -u scripts/run_ablation.py --out
results/ablation_raw_outputs_roundL.csv` — re-running the full 4-config English ablation
(bm25_only/full_hybrid/vector_only/no_retrieval, 200 queries each = 800 generations) now that the
retriever fix (item 4 above) is baked in. This changes bm25_only and full_hybrid's behavior
(vector_only is untouched since exact-match only fires when lambda>0). Log:
`results/roundL_ablation_log.txt`. **This job WAS stuck once already** (a duplicate/leftover
process from an earlier attempt in this session was fighting it for the same output file and
system RAM — killed both, confirmed memory recovered 1.81GB→3.57GB free, relaunched clean). If
you resume and this looks stuck again (no new log lines for >2min, python CPU time not
increasing), check `Get-Process python` and `Get-CimInstance Win32_Process | Where CommandLine
-like "*run_ablation.py*"` for duplicates before assuming it's a new problem.

**IMPORTANT resume check**: if `b1a243mbl` finished while paused, read
`results/roundL_ablation_log.txt` tail first — if it says "Wrote 800 rows to
results/ablation_raw_outputs_roundL.csv" it completed cleanly; run
`python scripts/compute_metrics.py --raw results/ablation_raw_outputs_roundL.csv` (check the
script's actual CLI flags first, don't guess) to get the summary table, matching the pattern of
every earlier round's `_metrics_summary_*.csv`/`_metrics_per_query_*.csv` outputs.

## Queued next (in priority order, not yet started)

1. Once Round L (English ablation) finishes: compute metrics, then run the novel pipeline
   **twice** more — reranker OFF (Round L baseline-equivalent for the adaptive pipeline) and
   reranker ON with the NEW fine-tuned reranker (`use_reranker=True` now loads
   `models/finetuned_reranker_domain/` automatically) — compare both against Round L's
   full_hybrid to see whether fine-tuning flipped the reranker from a significant negative to
   neutral-or-positive. This is the single most interesting open question left.
2. Re-run the Banglish ablation + novel pipeline with the retriever fix (likely also improves
   Banglish entity-heavy queries, though Banglish faculty-lookup phrasing wasn't specifically
   checked against the new faculty index — worth spot-checking).
3. Run `data/test_queries_faculty.csv` (50 queries) through the full novel pipeline, score with
   the 4 metrics — closes the FacultyAvailability coverage gap.
4. Faithfulness gap diagnosis (adaptive_novel 0.856 vs baselines 0.929-0.943): investigated
   already (see below) — inconclusive on the easy hypotheses, needs either acceptance as a
   genuine open finding or a second LLM-judge pass for inter-judge agreement (was on the todo
   list, not started: use a *different* Ollama model, if one is pulled, or at minimum a
   differently-worded judge prompt, as a partial substitute for real human annotation — NOT a
   replacement, be honest about that distinction in the paper).
5. Re-run faithfulness scoring on the new Round L outputs once available (old faithfulness
   numbers are from the pre-fix retriever).
6. Rewrite `paper.tex` to incorporate ALL of this: 7-table corpus (fix the FacultyList omission),
   the exact-match fix + before/after table, the cross-lingual stress test as a new subsection
   (replacing the old "why the null results happen" hedge with an actual positive validation),
   updated abstention calibration table, FacultyAvailability results, GPU mention (remove the old
   "CPU-only" limitation), reranker fine-tuning result once Round L reranker-on is in. Regenerate
   `fig_dataset_dist.png` with 7 bars (I already have the corrected script at
   `C:\Users\manxa\AppData\Local\Temp\claude\...\scratchpad\make_dataset_fig.py` — labels/counts
   already fixed to include FacultyList=223, just needs re-running once memory isn't tight, it
   failed twice on `OpenBLAS error: Memory allocation` while Round L was contending for RAM).
   Recompile the PDF after.
7. Final honest summary to the user covering what got fixed vs. what remains a genuine, disclosed
   limitation (human hallucination annotation still cannot be done by me — flagged to the user
   already, they acknowledged this upfront).

## Faithfulness gap investigation notes (partial, not concluded)

Checked: word count (novel 21.9 vs baselines 20.9-23.1, NOT longer — ruled out "more verbose"
hypothesis), route (entity_heavy 0.88 vs open_ended 0.84, small gap), graph_augmented (0.875 n=4
vs 0.854 n=46, too small to matter), exact_match_any (0.88 vs 0.84). None fully explain the
~0.07-0.09 gap vs baselines. Qualitative read of the 11 lowest-scoring rows (of 50, all <0.7):
recurring pattern is the LLM stating an additional plausible claim beyond what's literally in one
retrieved chunk (e.g. a "chain" answer restating a graph block's arrow-notation as prose, which
the automated judge sometimes doesn't credit as "explicitly stated" even though the info is
genuinely there) — a real, disclosable, non-fabricated finding, but not something I fixed. Report
honestly as "investigated, partially explained, root cause not fully isolated" rather than either
"unexplained" (stale, since I did investigate) or a false confident fix.

## Environment reminders (things that bit me this segment, don't re-learn them the hard way)

- `. .\scripts\env_setup.ps1` FIRST in every PowerShell call, or you get the old CPU-only
  WindowsApps python and cache/temp dirs pointing at C: again.
- System RAM is genuinely tight (15.73GB total) once Ollama's model (llama-server, ~3.5GB
  resident) + the GPU venv's embedding model + a generation script are all live at once — leave
  headroom, don't stack multiple heavy background jobs.
- If a background job that talks to Ollama fails with `unable to allocate CUDA_Host buffer`,
  it's VRAM contention (only 4GB total) — don't run GPU fine-tuning and Ollama generation at the
  same time; they take turns, not parallel.
- `sentence-transformers` must stay at **5.6.1** (matches what the fine-tuned models were saved
  with) — if a `ModuleNotFoundError: sentence_transformers.base` reappears, it means something
  downgraded it again.
- Always check for **duplicate/leftover background processes** before concluding a job is
  "stuck" — `Get-CimInstance Win32_Process | Where-Object CommandLine -like "*<script>*"` — this
  cost real time once already this segment.

=== OLDER, SUPERSEDED CONTEXT (kept for history, not the current resume point) ===

Everything below this line describes state from BEFORE this segment (rounds A-K, the original
paper.tex rewrite around the novel pipeline, Banglish evaluation, embedding fine-tuning
(Top-1 0.185→0.256 pre-Round-L-retriever-fix numbers — since superseded), reranker-off decision
pre-fine-tuning, etc.). It's still accurate AS HISTORY but every number in it that overlaps with
the "big-picture wins" section above should be treated as superseded by this segment's rework,
not read as the current state.

Round K resume mechanics, round-by-round chronology (A-K), and other pre-this-segment detail
previously lived here and remain available via `git`-free history is not applicable (no git in
this repo) — the full prior detail is preserved in the conversation transcript if ever needed,
not duplicated here to keep this file focused on what's actually still actionable.
