"""
Consistency check: same query run N times, measuring variance in the
retrieved top-1 document and the generated answer -- a genuinely new
measurement identified against an external "A* evaluation framework"
checklist, not previously done in this project.

Motivated directly by two already-documented facts elsewhere in this
project: (1) generation uses temperature=0.0 with a fixed seed=42
everywhere (pipeline/ollama_client.py), which should in principle make
generation fully deterministic; (2) the paper's own reranker-route
-conditional section already reports a SMALL, previously-found source of
non-determinism (GPU-inference-level floating-point non-determinism
producing minor paraphrase-level differences, e.g. "MAT216 (HP) -
MAT120 - MAT110" vs. "MAT216-MAT120-MAT110" for the identical retrieved
context) found incidentally while debugging an unrelated cross-process
bug. This script measures that directly, at a proper sample size, rather
than relying on an incidental anecdote.

Samples N_QUERIES real test queries (a mix of entity-heavy and
open-ended, by is_entity_heavy), runs each through the deployed
NovelPipeline N_REPEATS times (retrieval + generation both), and reports:
  - retrieval consistency: is the same top-1 doc_id returned every time?
  - generation consistency: is the exact same answer string produced
    every time?
  - for generation mismatches, a normalized edit-distance-based
    similarity (difflib) quantifying HOW different the repeats are,
    since "not byte-identical" and "substantively different answer" are
    not the same finding.

Usage: python scripts/measure_consistency.py
"""

import difflib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate as generate_answer

OUT_PATH = ROOT / "results_final" / "robustness" / "consistency_check.csv"
OUT_SUMMARY_PATH = ROOT / "results_final" / "robustness" / "consistency_summary.json"
N_QUERIES = 20
N_REPEATS = 3
SAMPLE_SEED = 42


def main():
    queries_df = pd.read_csv(ROOT / "data" / "test_queries.csv")
    entity_heavy = queries_df[queries_df["is_entity_heavy"] == True].sample(
        n=min(10, (queries_df["is_entity_heavy"] == True).sum()), random_state=SAMPLE_SEED)
    open_ended = queries_df[queries_df["is_entity_heavy"] == False].sample(
        n=min(10, (queries_df["is_entity_heavy"] == False).sum()), random_state=SAMPLE_SEED)
    sample = pd.concat([entity_heavy, open_ended]).reset_index(drop=True)
    print(f"Sampled {len(sample)} queries ({len(entity_heavy)} entity-heavy, {len(open_ended)} open-ended)", flush=True)

    pipeline = NovelPipeline()
    retriever = HybridRetriever()

    rows = []
    t0 = time.perf_counter()
    for i, row in sample.iterrows():
        query = row["query"]
        top1_ids = []
        answers = []
        for rep in range(N_REPEATS):
            results, _ = retriever.retrieve_adaptive(query)
            top1_ids.append(results[0]["doc_id"] if results else None)
            answer_text, meta, context_used, generation_s = pipeline.answer(query, generate_answer)
            answers.append(answer_text)
        retrieval_consistent = len(set(top1_ids)) == 1
        exact_match_all = len(set(answers)) == 1
        if not exact_match_all:
            sims = []
            for a in range(1, N_REPEATS):
                sims.append(difflib.SequenceMatcher(None, answers[0], answers[a]).ratio())
            min_sim = min(sims)
        else:
            min_sim = 1.0
        rows.append({
            "query_id": row["query_id"], "is_entity_heavy": row["is_entity_heavy"],
            "retrieval_consistent": retrieval_consistent,
            "exact_match_all_repeats": exact_match_all, "min_pairwise_similarity": min_sim,
            "answers": json.dumps(answers),
        })
        print(f"  {i+1}/{len(sample)} ({row['query_id']}): retrieval_consistent={retrieval_consistent} "
              f"exact_match={exact_match_all} min_sim={min_sim:.3f} "
              f"({time.perf_counter()-t0:.1f}s elapsed)", flush=True)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n = len(out)
    n_exact = int(out["exact_match_all_repeats"].sum())
    n_retrieval_consistent = int(out["retrieval_consistent"].sum())
    summary = {
        "n_queries": n, "n_repeats": N_REPEATS,
        "n_retrieval_consistent": n_retrieval_consistent,
        "fraction_retrieval_consistent": round(n_retrieval_consistent / n, 4) if n else 0.0,
        "n_exact_match_all_repeats": n_exact,
        "fraction_exact_match": round(n_exact / n, 4) if n else 0.0,
        "mean_min_pairwise_similarity": round(float(out["min_pairwise_similarity"].mean()), 4),
        "min_min_pairwise_similarity": round(float(out["min_pairwise_similarity"].min()), 4),
    }
    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRetrieval: {n_retrieval_consistent}/{n} queries returned the same top-1 doc_id across all repeats")
    print(f"Generation: {n_exact}/{n} queries produced byte-identical answers across all {N_REPEATS} repeats "
          f"({100*n_exact/n:.1f}%)")
    print(f"Mean min-pairwise-similarity (1.0 = identical): {summary['mean_min_pairwise_similarity']:.4f}")
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
