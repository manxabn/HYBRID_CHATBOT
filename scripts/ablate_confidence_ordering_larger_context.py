"""
Re-tests zig-zag confidence-ordered context assembly (scripts/ablate_
confidence_ordering.py found a null result at final_k=5, n=98) at a LARGER
context size (final_k=10, double the default) to test the specific
hypothesis proposed for why it came back null: this pipeline's contexts
(2-5 chunks) may simply be too small for the Lost-in-the-Middle position
effect to have anything to fix, since that effect was originally measured
with 10-20+ retrieved documents (Liu et al. 2024) and Jin et al.'s 2025
zig-zag mechanism is reported to help MORE as the number of retrieved
passages grows.

Same method as the original ablation (paired, same eligible-query filter,
same MIN_PIECES>=3 spirit) but with NovelPipeline(final_k=10) instead of
the default 5, so there is actually enough "middle" for the mechanism to
plausibly matter. If this ALSO comes back null, that is a stronger,
more specific negative finding (ruling out "context too short" as the
explanation) rather than a second untested guess.

Usage: python scripts/ablate_confidence_ordering_larger_context.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate

QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "confidence_ordering_ablation_larger_context_raw.csv"
FINAL_K = 10


def main():
    df = pd.read_csv(QUERIES_PATH)

    probe = NovelPipeline(final_k=FINAL_K)
    eligible = []
    for _, r in df.iterrows():
        _, meta = probe.build_context(r["query"])
        if meta["route"] == "open_ended" and not meta.get("abstain"):
            eligible.append(r["query_id"])
    chain_queries = df[df["query_id"].isin(eligible)].reset_index(drop=True)
    print(f"Open-ended, non-abstained queries at final_k={FINAL_K}: {len(chain_queries)}/{len(df)}")

    rows = []
    for use_ordering in [True, False]:
        pipeline = NovelPipeline(use_confidence_ordering=use_ordering, final_k=FINAL_K)
        label = "ordering_on" if use_ordering else "ordering_off"
        for _, r in chain_queries.iterrows():
            answer, meta, context, generation_s = pipeline.answer(r["query"], generate)
            n_pieces = len([p for p in (context or "").split("\n\n") if p.strip()])
            rows.append({
                "query_id": r["query_id"], "config": label, "query": r["query"],
                "reference_answer": r["reference_answer"], "generated_answer": answer,
                "abstained": meta["abstain"], "n_context_pieces": n_pieces,
            })
            print(f"  [{label}] {r['query_id']}: abstained={meta['abstain']} pieces={n_pieces}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"Mean context pieces: {out['n_context_pieces'].mean():.2f} "
          f"(original ablation at final_k=5 presumably averaged fewer)")


if __name__ == "__main__":
    main()
