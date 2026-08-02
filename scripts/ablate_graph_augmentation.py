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

This n=12 result is historical -- superseded by the n=31 expansion in
scripts/ablate_graph_augmentation_expanded.py, which is what paper.tex's
Table~\ref{tab:graph-ablation} now reports as the confirmed result. Kept
runnable for reference; --out now required to default-protect the
original n=12 output from being silently overwritten by a re-run (2026-
08-01 fix: this script previously hardcoded its output path, unlike every
other ablation script in this project, which follows run_ablation.py's
convention of an --out flag precisely to prevent this).

Usage: python scripts/ablate_graph_augmentation.py [--out PATH]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate
from pipeline.prerequisite_graph import PrerequisiteGraph

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
DEFAULT_OUT_PATH = ROOT / "results" / "graph_ablation_raw.csv"


def main(out_path: Path):
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
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH,
                         help=f"Output CSV path (default: {DEFAULT_OUT_PATH}, the original n=12 "
                              "historical result -- pass a different path to avoid overwriting it)")
    args = parser.parse_args()
    main(args.out)
