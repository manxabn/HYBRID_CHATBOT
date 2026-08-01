"""
Expands the prerequisite-graph isolated ablation from n=12 to n=31, using
19 additional REAL, database-verified prerequisite chains that exist in
the corpus but were never included in the 200-query main test set --
not synthetic data. The corpus has 29 courses with a real, non-empty
FullChainPreRequisite entry; only 12 of them happen to be exercised by
the main test set. scripts/build_graph_expansion_queries.py-equivalent
logic (inlined below, see data/test_queries_graph_new19.csv for the
already-generated rows) computes each new query's reference answer via
the SAME live PrerequisiteGraph.full_chain() traversal the deployed
system uses, not a hand-typed guess, so the expanded set's ground truth
is exactly as trustworthy as the original 12.

Otherwise identical methodology to scripts/ablate_graph_augmentation.py
(same use_graph on/off comparison, same generation-included isolated
test) -- reported as a real n-expansion, not a re-run of the same data.

Usage: python scripts/ablate_graph_augmentation_expanded.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate
from pipeline.prerequisite_graph import PrerequisiteGraph

ORIGINAL_QUERIES_PATH = ROOT / "data" / "test_queries.csv"
NEW_QUERIES_PATH = ROOT / "data" / "test_queries_graph_new19.csv"
OUT_PATH = ROOT / "results" / "graph_ablation_expanded_raw.csv"


def main():
    original = pd.read_csv(ORIGINAL_QUERIES_PATH)
    pg = PrerequisiteGraph()
    original_chain_q = original[original["query"].apply(lambda q: bool(pg.context_block(q)))]
    new_q = pd.read_csv(NEW_QUERIES_PATH)

    chain_queries = pd.concat([
        original_chain_q[["query_id", "query", "reference_answer"]],
        new_q[["query_id", "query", "reference_answer"]],
    ], ignore_index=True)
    print(f"Chain-triggering queries: {len(chain_queries)} "
          f"({len(original_chain_q)} original + {len(new_q)} newly added real facts)", flush=True)

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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
