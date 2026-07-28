"""
Run all four retrieval configs against data/test_queries.csv and save every
raw output to results/ablation_raw_outputs.csv, with per-query retrieval and
generation timing captured as a byproduct (not a substitute for a dedicated
latency study -- this is a single run, not reconciled against network
variance etc.).

Usage:
    python scripts/run_ablation.py --smoke               # first 5 queries only
    python scripts/run_ablation.py                       # full run, linear fusion (default, matches existing results/)
    python scripts/run_ablation.py --sample-size 20       # stratified random subset (fixed seed=42, keeps
        # the is_entity_heavy mix proportional) -- for when the full 200-query
        # run isn't time-feasible (measured on this machine: CPU-only Ollama
        # inference averages ~25-47s/generation, so the full 800-generation
        # run is a 5.5-10hr job, not something to run casually). Written to
        # its own results/ablation_raw_outputs_n{N}.csv by default so a
        # partial run can never masquerade as, or collide with, the full run.
    python scripts/run_ablation.py --fusion rrf --out results/ablation_raw_outputs_rrf.csv
        # novelty run: RRF fusion + course-code normalization + alias resolution
        # (see pipeline/hybrid_retriever.py docstring). Written to a separate
        # file by default so it never overwrites the linear-fusion baseline.
"""

import argparse
import csv
import json
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
OUT_PATH = ROOT / "results" / "ablation_raw_outputs.csv"
META_PATH = ROOT / "results" / "run_metadata.json"

CONFIGS = [
    ("full_hybrid", 0.5),
    ("bm25_only", 1.0),
    ("vector_only", 0.0),
    ("no_retrieval", None),
]

FIELDNAMES = [
    "query_id", "config", "lambda", "query", "reference_answer",
    "retrieved_context", "generated_answer",
    "retrieval_s", "generation_s", "total_s", "timestamp",
    "query_confidence",
]


SAMPLE_SEED = 42


def run(smoke: bool, fusion: str, out_path: Path, sample_size: int = None, queries_path: Path = None):
    df = pd.read_csv(queries_path or QUERIES_PATH)
    if smoke:
        df = df.head(5)
    elif sample_size is not None:
        # Stratified by is_entity_heavy so a small sample doesn't accidentally
        # skew all-open-ended or all-entity-heavy; fixed seed so re-running
        # with the same --sample-size is reproducible.
        df = (
            df.groupby("is_entity_heavy", group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), max(1, round(sample_size * len(g) / len(df)))),
                random_state=SAMPLE_SEED,
            ))
        )

    retriever = HybridRetriever(fusion=fusion)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or smoke
    mode = "w" if write_header else "a"

    start_time = datetime.now(timezone.utc).isoformat()
    n_rows = 0

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for config_name, lam in CONFIGS:
            for _, row in df.iterrows():
                query = row["query"]
                ref = row["reference_answer"]

                t0 = time.perf_counter()
                context = None
                retrieved_docs = []
                confidence = ""
                if lam is not None:
                    retrieved_docs = retriever.retrieve(query, lam)
                    context = "\n\n".join(d["text"] for d in retrieved_docs)
                    if retrieved_docs:
                        confidence = f"{retrieved_docs[0]['query_confidence']:.4f}"
                t1 = time.perf_counter()
                retrieval_s = t1 - t0

                t2 = time.perf_counter()
                answer = generate(query, context)
                t3 = time.perf_counter()
                generation_s = t3 - t2

                writer.writerow({
                    "query_id": row["query_id"],
                    "config": config_name,
                    "lambda": lam if lam is not None else "",
                    "query": query,
                    "reference_answer": ref,
                    "retrieved_context": context or "",
                    "generated_answer": answer,
                    "retrieval_s": f"{retrieval_s:.4f}",
                    "generation_s": f"{generation_s:.4f}",
                    "total_s": f"{retrieval_s + generation_s:.4f}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query_confidence": confidence,
                })
                f.flush()
                n_rows += 1
                print(f"[{config_name}] {row['query_id']}: {generation_s:.2f}s gen, "
                      f"{retrieval_s:.3f}s retrieval")

    end_time = datetime.now(timezone.utc).isoformat()

    if not smoke:
        meta = {
            "model": MODEL,
            "fusion": fusion,
            "n_queries": len(df),
            "is_full_run": sample_size is None,
            "sample_size_requested": sample_size,
            "sample_seed": SAMPLE_SEED if sample_size is not None else None,
            "n_configs": len(CONFIGS),
            "n_rows_written": n_rows,
            "configs": [{"name": c, "lambda": l} for c, l in CONFIGS],
            "start_time_utc": start_time,
            "end_time_utc": end_time,
        }
        # Preserve the original results/run_metadata.json name for the default
        # linear run so it stays exactly where existing tooling expects it;
        # any other (out_path, fusion) combination gets its own metadata file.
        meta_path = META_PATH if out_path == OUT_PATH else out_path.parent / f"{out_path.stem}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metadata to {meta_path}")

    print(f"Wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run only 5 queries as a smoke test")
    parser.add_argument("--sample-size", type=int, default=None,
                         help="Run a stratified random subset of this many queries instead of the "
                              "full 200 (fixed seed=42, keeps is_entity_heavy proportion). Use when "
                              "the full 800-generation run isn't time-feasible.")
    parser.add_argument("--fusion", choices=["linear", "rrf"], default="linear",
                         help="Retrieval fusion mode (default: linear, matches existing results/)")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output CSV path (default: results/ablation_raw_outputs.csv for a full "
                              "linear run; results/ablation_raw_outputs_rrf.csv for a full rrf run; "
                              "results/ablation_raw_outputs_n{N}[_rrf].csv when --sample-size is given "
                              "-- a sampled run never writes to the full-run path, so partial results "
                              "can't masquerade as, or overwrite, the complete run)")
    parser.add_argument("--queries", type=Path, default=None,
                         help="Override the input queries CSV (default: data/test_queries.csv). "
                              "e.g. data/test_queries_banglish.csv")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        if args.sample_size is not None:
            suffix = f"_n{args.sample_size}" + ("_rrf" if args.fusion == "rrf" else "")
            out_path = ROOT / "results" / f"ablation_raw_outputs{suffix}.csv"
        else:
            out_path = OUT_PATH if args.fusion == "linear" else ROOT / "results" / "ablation_raw_outputs_rrf.csv"

    run(smoke=args.smoke, fusion=args.fusion, out_path=out_path, sample_size=args.sample_size,
        queries_path=args.queries)
