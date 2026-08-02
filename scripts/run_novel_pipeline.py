"""
Run the novel pipeline (adaptive fusion routing + cross-encoder reranking +
prerequisite-graph augmentation + confidence-gated abstention,
pipeline/novel_pipeline.py) against data/test_queries.csv, writing raw
outputs to their own file so this never touches the existing baseline
results/ CSVs.

Usage:
    python scripts/run_novel_pipeline.py --smoke
    python scripts/run_novel_pipeline.py --sample-size 40
    python scripts/run_novel_pipeline.py                       # full 200
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

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate, MODEL

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
CONFIG_NAME = "adaptive_novel"

FIELDNAMES = [
    "query_id", "config", "query", "reference_answer",
    "retrieved_context", "generated_answer",
    "route", "fusion", "lambda", "abstained", "raw_abstain",
    "sufficient_context_override", "graph_augmented", "faculty_room_augmented", "exact_match_any", "question_match_any",
    "translated_query", "normalized_query",
    "retrieval_s", "rerank_s", "generation_s", "total_s",
    "timestamp", "query_confidence", "query_top1_score",
]

SAMPLE_SEED = 42


def run(smoke: bool, out_path: Path, sample_size: int = None, queries_path: Path = None,
        use_reranker: bool = False, use_query_translation: bool = False, rerank_pool_size: int = 10,
        use_entity_normalization: bool = False, use_faculty_room_lookup: bool = False,
        rerank_route: str = "all"):
    df = pd.read_csv(queries_path or QUERIES_PATH)
    if smoke:
        df = df.head(5)
    elif sample_size is not None:
        df = (
            df.groupby("is_entity_heavy", group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), max(1, round(sample_size * len(g) / len(df)))),
                random_state=SAMPLE_SEED,
            ))
        )

    pipeline = NovelPipeline(use_reranker=use_reranker, use_query_translation=use_query_translation,
                              rerank_pool_size=rerank_pool_size, use_entity_normalization=use_entity_normalization,
                              use_faculty_room_lookup=use_faculty_room_lookup, rerank_route=rerank_route)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc).isoformat()
    n_rows = 0
    n_abstained = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for _, row in df.iterrows():
            query = row["query"]
            ref = row["reference_answer"]

            t0 = time.perf_counter()
            answer, meta, context, generation_s = pipeline.answer(query, generate)
            t1 = time.perf_counter()

            if meta["abstain"]:
                n_abstained += 1

            writer.writerow({
                "query_id": row["query_id"],
                "config": CONFIG_NAME,
                "query": query,
                "reference_answer": ref,
                "retrieved_context": context or "",
                "generated_answer": answer,
                "route": meta["route"],
                "fusion": meta["fusion"],
                "lambda": meta["lambda"],
                "abstained": meta["abstain"],
                "raw_abstain": meta["raw_abstain"],
                "sufficient_context_override": meta["sufficient_context_override"],
                "graph_augmented": meta["graph_augmented"],
                "faculty_room_augmented": meta["faculty_room_augmented"],
                "exact_match_any": meta["exact_match_any"],
                "question_match_any": meta["question_match_any"],
                "translated_query": meta["translated_query"],
                "normalized_query": meta["normalized_query"],
                "retrieval_s": f"{meta['retrieval_s']:.4f}",
                "rerank_s": f"{meta['rerank_s']:.4f}",
                "generation_s": f"{generation_s:.4f}",
                "total_s": f"{t1 - t0:.4f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_confidence": f"{meta['query_confidence']:.4f}",
                "query_top1_score": f"{meta['query_top1_score']:.4f}",
            })
            f.flush()
            n_rows += 1
            print(f"[{meta['route']}/{meta['fusion']}] {row['query_id']}: "
                  f"{generation_s:.2f}s gen, {meta['retrieval_s']:.3f}s retrieval, "
                  f"{meta['rerank_s']:.3f}s rerank"
                  + (" [ABSTAINED]" if meta["abstain"] else ""))

    end_time = datetime.now(timezone.utc).isoformat()

    if not smoke:
        meta_path = out_path.parent / f"{out_path.stem}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL,
                "config_name": CONFIG_NAME,
                "n_queries": len(df),
                "n_rows_written": n_rows,
                "n_abstained": n_abstained,
                "is_full_run": sample_size is None,
                "sample_size_requested": sample_size,
                "sample_seed": SAMPLE_SEED if sample_size is not None else None,
                "start_time_utc": start_time,
                "end_time_utc": end_time,
            }, f, indent=2)
        print(f"Wrote metadata to {meta_path}")

    print(f"Wrote {n_rows} rows to {out_path} ({n_abstained} abstained)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=None,
                         help="Override the input queries CSV (default: data/test_queries.csv).")
    parser.add_argument("--use-reranker", action="store_true",
                         help="Re-enable the cross-encoder reranker (OFF by default as of "
                              "2026-07-27 -- confirmed via ablation to be a net negative now "
                              "that fine-tuned embeddings make initial ranking strong; this flag "
                              "restores the old behavior, e.g. to test a future fine-tuned reranker)")
    parser.add_argument("--query-translation", action="store_true",
                         help="Enable cross-lingual query-translation retrieval for Banglish "
                              "queries (OFF by default as of 2026-07-27 -- tested in isolation on "
                              "the Banglish eval set and found not statistically significant "
                              "either direction, p>0.46 on all 4 metrics; the fine-tuned "
                              "embeddings already bridge the cross-lingual gap on their own, so "
                              "translation's extra LLM-call latency isn't justified as a default)")
    parser.add_argument("--rerank-pool-size", type=int, default=10,
                         help="Candidate pool size the reranker re-sorts before top-final_k "
                              "(default 10, only relevant with --use-reranker). Magomere et al. "
                              "(2025, ACL Findings, arXiv:2503.03417) find reranking a large "
                              "candidate set can inject noise that hurts an already-strong first-"
                              "stage ranking, and recommend a smaller pool as a mitigation -- "
                              "pass e.g. 3 or 5 to test that directly on this pipeline.")
    parser.add_argument("--use-faculty-room-lookup", action="store_true",
                         help="Enable the course->instructor->office-room cross-reference lookup "
                              "(OFF by default as of 2026-07-31 -- new component, not yet default "
                              "pending its own isolated ablation, same practice as --use-reranker).")
    parser.add_argument("--rerank-route", choices=["all", "open_ended"], default="all",
                         help="Which route(s) --use-reranker applies to (default: all, existing "
                              "behavior). 'open_ended' restricts reranking to non-entity-heavy "
                              "queries, motivated by the existing reranker-on run's own per-route "
                              "breakdown showing real damage concentrated in entity-heavy queries "
                              "(where reranking fights the exact-match ceiling) vs. near-zero, "
                              "mixed-sign effect on open-ended queries -- see pipeline/novel_pipeline.py.")
    parser.add_argument("--entity-normalization", action="store_true",
                         help="Enable the entity-normalization retrieval fallback (fuzzy/LLM "
                              "correction of malformed course codes and misspelled names). NOTE: "
                              "NovelPipeline itself defaults this ON (validated 2026-07-28, 8/8 "
                              "on the malformed-query ablation -- see pipeline/novel_pipeline.py), "
                              "but THIS script passes False unless this flag is given; the script "
                              "default is kept OFF so re-runs stay comparable with earlier CSVs "
                              "launched the same way, not because the feature is unvalidated.")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        suffix = f"_n{args.sample_size}" if args.sample_size is not None else ""
        suffix += "_reranker" if args.use_reranker else ""
        suffix += f"_pool{args.rerank_pool_size}" if args.use_reranker and args.rerank_pool_size != 10 else ""
        suffix += "_rerankopenended" if args.use_reranker and args.rerank_route == "open_ended" else ""
        suffix += "_translation" if args.query_translation else ""
        suffix += "_entitynorm" if args.entity_normalization else ""
        out_path = ROOT / "results" / f"novel_pipeline_raw_outputs{suffix}.csv"

    run(smoke=args.smoke, out_path=out_path, sample_size=args.sample_size, queries_path=args.queries,
        use_reranker=args.use_reranker, use_query_translation=args.query_translation,
        rerank_pool_size=args.rerank_pool_size, use_entity_normalization=args.entity_normalization,
        use_faculty_room_lookup=args.use_faculty_room_lookup, rerank_route=args.rerank_route)
