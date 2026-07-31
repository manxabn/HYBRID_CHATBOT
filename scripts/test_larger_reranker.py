"""
Tests whether a substantially LARGER, still-local, open-weight reranker
(BAAI/bge-reranker-v2-m3, ~568M params -- the same model BanglAssist
itself uses, verified via direct paper fetch earlier this session) does
better than this project's existing small reranker (33M-param MS-MARCO
MiniLM), which was consistently negative across every prior ablation.

Motivated directly by a literature check (arXiv:2604.01733): a comparable
paper's reranker win came from a much larger commercial reranker (Cohere
Rerank v4.0 Pro), not from reranking in general -- this tests whether
capacity, not the reranking approach itself, was the missing ingredient,
using a real open-weight model instead of a proprietary API this
project's local-model design doesn't use.

Stratified n=40 sample (matching the faculty_room_lookup ablation's
scale), real Ollama generations, reranker-off (full_hybrid) as the
baseline -- same methodology as every other reranker ablation this
project has run.

Usage: python scripts/test_larger_reranker.py
"""

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.reranker import Reranker
from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "novel_pipeline_raw_outputs_bge_reranker_v2m3.csv"
SAMPLE_SIZE = 40
SAMPLE_SEED = 42
FIELDNAMES = ["query_id", "config", "query", "reference_answer", "retrieved_context",
              "generated_answer", "abstained"]


def main():
    df = pd.read_csv(QUERIES_PATH)
    df = (
        df.groupby("is_entity_heavy", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), max(1, round(SAMPLE_SIZE * len(g) / len(df)))),
            random_state=SAMPLE_SEED,
        ))
    )
    print(f"Sampled {len(df)} queries", flush=True)

    # Resume support: skip query_ids already written, in case of a crash
    # mid-run (this model is large enough on this 4GB card that a crash
    # partway through is a real, not hypothetical, risk).
    completed = set()
    write_header = True
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        prior = pd.read_csv(OUT_PATH)
        completed = set(prior["query_id"])
        write_header = False
        print(f"Resuming: {len(completed)} query_ids already done", flush=True)

    print("Loading BAAI/bge-reranker-v2-m3...", flush=True)
    reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
    pipeline = NovelPipeline(reranker=reranker, use_reranker=True)

    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        for _, row in df.iterrows():
            if row["query_id"] in completed:
                continue
            query = row["query"]
            answer, meta, context, generation_s = pipeline.answer(query, generate)
            writer.writerow({
                "query_id": row["query_id"], "config": "adaptive_novel_bge_v2m3",
                "query": query, "reference_answer": row["reference_answer"],
                "retrieved_context": context or "", "generated_answer": answer,
                "abstained": meta["abstain"],
            })
            f.flush()
            print(f"[{meta['route']}] {row['query_id']}: {generation_s:.2f}s"
                  + (" [ABSTAINED]" if meta["abstain"] else ""), flush=True)

    final = pd.read_csv(OUT_PATH)
    print(f"\nWrote {len(final)} total rows to {OUT_PATH} ({final['abstained'].sum()} abstained)")


if __name__ == "__main__":
    main()
