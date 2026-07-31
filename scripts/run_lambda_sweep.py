"""
CLAUDE.md Task #3: finer lambda sweep, lambda in {0, 0.1, ..., 1.0}, to see
whether some blend point genuinely peaks above both single-method endpoints,
or whether performance is flat/monotonic between them.

Uses a fixed 60-query stratified subset (30 entity-heavy + 30 open-ended,
sampled with a fixed seed from the same data/test_queries.csv used for the
main 4-config ablation) rather than the full 200 -- 11 lambda points x 200
queries (2200 generations) was judged not worth the ~2.5hr runtime for a
sensitivity sweep whose job is to show a trend curve, not restate the main
comparison at full N. This trade-off is disclosed here, not hidden.

Also checks whether the optimal lambda differs between entity-heavy and
open-ended queries (CLAUDE.md Task #3's second question).
"""

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import generate, MODEL

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "lambda_sweep_raw_outputs.csv"
META_PATH = ROOT / "results" / "lambda_sweep_metadata.json"

SEED = 123
N_PER_STRATUM = 30
LAMBDAS = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0

FIELDNAMES = [
    "query_id", "lambda", "query", "reference_answer", "is_entity_heavy",
    "retrieved_context", "generated_answer", "retrieval_s", "generation_s", "total_s", "timestamp",
]


def build_subset():
    df = pd.read_csv(QUERIES_PATH)
    entity = df[df.is_entity_heavy].sample(n=N_PER_STRATUM, random_state=SEED)
    open_ended = df[~df.is_entity_heavy].sample(n=N_PER_STRATUM, random_state=SEED)
    return pd.concat([entity, open_ended]).reset_index(drop=True)


def main():
    subset = build_subset()
    retriever = HybridRetriever()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Resumable + flushed (2026-07-31, same fix applied to run_ablation.py
    # the same day after a real incident: a headerless-file bug from
    # write_header only checking existence, and unflushed progress prints
    # making a genuinely-working long job look stalled in its log).
    completed = set()
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        try:
            prior = pd.read_csv(OUT_PATH)
            if "query_id" not in prior.columns:
                raise ValueError("headerless")
        except (pd.errors.EmptyDataError, ValueError):
            prior = pd.read_csv(OUT_PATH, header=None, names=FIELDNAMES)
        completed = set(zip(prior["query_id"], prior["lambda"]))
        print(f"Resuming: {len(completed)} (query_id, lambda) pairs already done in {OUT_PATH}", flush=True)
    write_header = (not OUT_PATH.exists()) or OUT_PATH.stat().st_size == 0
    mode = "w" if write_header else "a"

    start_time = datetime.now(timezone.utc).isoformat()
    n_rows = 0

    with open(OUT_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for lam in LAMBDAS:
            for _, row in subset.iterrows():
                if (row["query_id"], lam) in completed:
                    continue
                query = row["query"]
                ref = row["reference_answer"]

                t0 = time.perf_counter()
                results = retriever.retrieve(query, lam)
                context = "\n\n".join(d["text"] for d in results)
                t1 = time.perf_counter()
                retrieval_s = t1 - t0

                t2 = time.perf_counter()
                answer = generate(query, context)
                t3 = time.perf_counter()
                generation_s = t3 - t2

                writer.writerow({
                    "query_id": row["query_id"],
                    "lambda": lam,
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
                print(f"[lambda={lam}] {row['query_id']}: {generation_s:.2f}s gen", flush=True)

    end_time = datetime.now(timezone.utc).isoformat()
    import json
    meta = {
        "model": MODEL,
        "n_queries_per_lambda": len(subset),
        "n_lambdas": len(LAMBDAS),
        "lambdas": LAMBDAS,
        "n_rows_written": n_rows,
        "seed": SEED,
        "start_time_utc": start_time,
        "end_time_utc": end_time,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {n_rows} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
