"""
Proper held-out validation of lambda=0.3 (the empirical peak found by
scripts/run_lambda_sweep.py's 60-query TUNING subset, 2026-07-31) against
the deployed lambda=0.5 and the single-method baselines.

Why this exists: picking whichever lambda already scores best on the same
data used to evaluate it is circular (this project's own standing
methodology note). This script fixes that by construction: it reconstructs
the EXACT 60-query tuning subset run_lambda_sweep.py used (same stratified
sample, same seed=123) and evaluates ONLY on the remaining 140 queries --
genuinely untouched by the lambda-selection process. The other four
configs (bm25_only/vector_only/full_hybrid/no_retrieval) were already
generated for all 200 queries in scripts/run_ablation.py's run
(results/ablation_raw_outputs.csv) -- this script generates ONLY the new
lambda=0.3 arm on the held-out 140, then compares against the EXISTING
baseline generations filtered to those same 140 query_ids, so the
comparison is apples-to-apples on a query set none of the four existing
configs needed to be re-run for.

Checkpointed/resumable + flushed prints from the start (lessons already
learned the hard way earlier the same day on run_ablation.py and
run_lambda_sweep.py).

Usage: python scripts/eval_lambda_held_out.py
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import generate, MODEL

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "lambda_0.3_held_out_raw.csv"
TUNE_SEED = 123
N_PER_STRATUM_TUNE = 30
LAMBDA = 0.3

FIELDNAMES = [
    "query_id", "config", "lambda", "query", "reference_answer", "is_entity_heavy",
    "retrieved_context", "generated_answer", "retrieval_s", "generation_s", "total_s", "timestamp",
]


def build_held_out_set():
    df = pd.read_csv(QUERIES_PATH)
    entity_tune = df[df.is_entity_heavy].sample(n=N_PER_STRATUM_TUNE, random_state=TUNE_SEED)
    open_tune = df[~df.is_entity_heavy].sample(n=N_PER_STRATUM_TUNE, random_state=TUNE_SEED)
    tune_ids = set(entity_tune["query_id"]) | set(open_tune["query_id"])
    held_out = df[~df["query_id"].isin(tune_ids)].reset_index(drop=True)
    return held_out, tune_ids


def main():
    held_out, tune_ids = build_held_out_set()
    print(f"Held-out set: {len(held_out)} queries (excludes the {len(tune_ids)}-query "
          f"tuning subset that selected lambda={LAMBDA})", flush=True)

    retriever = HybridRetriever()

    completed = set()
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        try:
            prior = pd.read_csv(OUT_PATH)
            if "query_id" not in prior.columns:
                raise ValueError("headerless")
        except (pd.errors.EmptyDataError, ValueError):
            prior = pd.read_csv(OUT_PATH, header=None, names=FIELDNAMES)
        completed = set(prior["query_id"])
        print(f"Resuming: {len(completed)} queries already done in {OUT_PATH}", flush=True)
    write_header = (not OUT_PATH.exists()) or OUT_PATH.stat().st_size == 0
    mode = "w" if write_header else "a"

    n_rows = 0
    with open(OUT_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for _, row in held_out.iterrows():
            if row["query_id"] in completed:
                continue
            query = row["query"]
            ref = row["reference_answer"]

            import time
            t0 = time.perf_counter()
            results = retriever.retrieve(query, LAMBDA)
            context = "\n\n".join(d["text"] for d in results)
            t1 = time.perf_counter()
            retrieval_s = t1 - t0

            t2 = time.perf_counter()
            answer = generate(query, context)
            t3 = time.perf_counter()
            generation_s = t3 - t2

            writer.writerow({
                "query_id": row["query_id"],
                "config": f"lambda_{LAMBDA}",
                "lambda": LAMBDA,
                "query": query,
                "reference_answer": ref,
                "is_entity_heavy": row["is_entity_heavy"],
                "retrieved_context": context,
                "generated_answer": answer,
                "retrieval_s": f"{retrieval_s:.4f}",
                "generation_s": f"{generation_s:.4f}",
                "total_s": f"{retrieval_s + generation_s:.4f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            f.flush()
            n_rows += 1
            print(f"[lambda={LAMBDA}] {row['query_id']}: {generation_s:.2f}s gen", flush=True)

    print(f"Wrote {n_rows} new rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
