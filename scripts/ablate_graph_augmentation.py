"""
Isolated ablation of prerequisite-graph augmentation (pipeline/novel_
pipeline.py's use_graph flag) -- flagged directly in paper.tex's pipeline-
stage summary table as implemented but never independently significance-
tested, unlike every other major component. Only 12/200 English test
queries actually trigger the graph block (full prerequisite CHAIN
questions, verified directly via PrerequisiteGraph.context_block), so this
runs the full pipeline (generation included) on exactly those 12 queries,
with use_graph=True vs. False, everything else held fixed (reranker off,
same retriever, same abstention gate).

Usage: python scripts/ablate_graph_augmentation.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate
from pipeline.prerequisite_graph import PrerequisiteGraph

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "graph_ablation_raw.csv"


def main():
    df = pd.read_csv(QUERIES_PATH)
    pg = PrerequisiteGraph()
    chain_queries = df[df["query"].apply(lambda q: bool(pg.context_block(q)))].reset_index(drop=True)
    print(f"Chain-triggering queries: {len(chain_queries)}/{len(df)}")

    rows = []
    for use_graph in [True, False]:
        pipeline = NovelPipeline(use_graph=use_graph)
        label = "graph_on" if use_graph else "graph_off"
        for _, r in chain_queries.iterrows():
            answer, meta, context, generation_s = pipeline.answer(r["query"], generate)
            rows.append({
                "query_id": r["query_id"], "config": label, "query": r["query"],
                "reference_answer": r["reference_answer"], "generated_answer": answer,
                "abstained": meta["abstain"], "graph_augmented": meta["graph_augmented"],
            })
            print(f"  [{label}] {r['query_id']}: abstained={meta['abstain']} graph_augmented={meta['graph_augmented']}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
