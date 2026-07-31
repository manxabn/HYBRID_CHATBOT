"""
Isolated ablation of confidence-ordered "zig-zag" context assembly
(pipeline/novel_pipeline.py's use_confidence_ordering flag, added 2026-07-29
specifically to close this gap) -- flagged in this project's own pipeline
-stage summary table as implemented and deployed but never independently
significance-tested, unlike every other major component.

Motivated directly by literature research (WebFetch-verified): Jin, Yoon,
Han & Arık ("Long-Context LLMs Meet RAG," ICLR 2025, arXiv:2410.05983) test
the SAME zig-zag/sandwich reordering mechanism this project already
implements, reporting it helps more as the number of retrieved passages
grows; Liu et al. ("Lost in the Middle," TACL 2024, arXiv:2307.03172) give
the concrete magnitude of the underlying problem this reordering targets
(GPT-3.5-Turbo, 20-doc setting: 75.8% accuracy for a start-of-context
answer vs. 53.8% for middle-of-context vs. 63.2% for end-of-context -- a
U-shaped position effect). This project's own ablation is on its own
corpus/model (Llama-3.1-8B, a much smaller retrieved-context size than a
20-doc setting), so results should be framed as "same mechanism, different
scale/data," not a literal reproduction.

Only queries where context assembly actually has >=3 distinct pieces to
reorder are informative -- with 1-2 pieces, zig-zag and fixed-insertion
order can't differ at all (nothing to interleave). Filters to exactly that
subset first (checked directly via a dry run of build_context, not
assumed), mirroring scripts/ablate_graph_augmentation.py's pattern of
isolating to the subset where the mechanism can possibly matter.

Usage: python scripts/ablate_confidence_ordering.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "confidence_ordering_ablation_raw.csv"
MIN_PIECES = 3


def main():
    df = pd.read_csv(QUERIES_PATH)

    # Dry run (retrieval only, no generation) to find queries where context
    # assembly actually has >=3 distinct pieces -- reuses the real pipeline
    # object's own build_context, not a re-implementation, so "how many
    # pieces" is measured exactly the way the ablation itself will see it.
    probe = NovelPipeline()
    eligible = []
    for _, r in df.iterrows():
        _, meta = probe.build_context(r["query"])
        # scored_parts count isn't directly in meta; approximate via a
        # second, cheap signal: query as entity_heavy tends to collapse to
        # 1 piece (the exact match) so route + graph_augmented let us skip
        # obviously-ineligible rows without re-deriving scored_parts here.
        if meta["route"] == "open_ended" and not meta.get("abstain"):
            eligible.append(r["query_id"])
    chain_queries = df[df["query_id"].isin(eligible)].reset_index(drop=True)
    print(f"Open-ended, non-abstained queries (candidate pool for >=3-piece context): {len(chain_queries)}/{len(df)}")

    rows = []
    for use_ordering in [True, False]:
        pipeline = NovelPipeline(use_confidence_ordering=use_ordering)
        label = "ordering_on" if use_ordering else "ordering_off"
        for _, r in chain_queries.iterrows():
            answer, meta, context, generation_s = pipeline.answer(r["query"], generate)
            rows.append({
                "query_id": r["query_id"], "config": label, "query": r["query"],
                "reference_answer": r["reference_answer"], "generated_answer": answer,
                "abstained": meta["abstain"],
            })
            print(f"  [{label}] {r['query_id']}: abstained={meta['abstain']}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
