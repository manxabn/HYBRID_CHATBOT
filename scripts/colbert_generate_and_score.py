"""
Step 2 of the ColBERT-as-retriever end-to-end generation-quality ablation
(step 1: scripts/colbert_retrieve_context.py, run in the isolated
.venv_colbert). Reads that script's retrieved-context CSV, calls the same
`generate()` used by scripts/run_ablation.py for every other config, and
writes a raw-outputs CSV in the same shape scripts/compute_metrics.py
already scores (BLEU/ROUGE-L/BERTScore/METEOR) -- so this row can be
scored with the exact same, unmodified scoring code as the BM25-only/
Vector-only/No-retrieval rows in Table~\ref{tab:ablation}, and the
resulting numbers are directly comparable.

Runs in the MAIN project venv (Ollama HTTP client lives there; pylate/CUDA
torch is not needed for this step).

Usage:
  python scripts/colbert_generate_and_score.py --label colbert_external
  python scripts/compute_metrics.py --raw results/ablation_raw_outputs_colbert_external.csv \\
      --per-query-out results/colbert_external_generation_metrics_per_query.csv \\
      --summary-out results/colbert_external_generation_metrics_summary.csv
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ollama_client import generate

FIELDNAMES = [
    "query_id", "config", "query", "reference_answer",
    "retrieved_context", "generated_answer", "generation_s", "timestamp",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="colbert_external")
    args = parser.parse_args()

    context_path = ROOT / "results" / f"{args.label}_retrieved_context.csv"
    out_path = ROOT / "results" / f"ablation_raw_outputs_{args.label}.csv"

    df = pd.read_csv(context_path)
    print(f"Generating for {len(df)} queries (config={args.label})...", flush=True)

    completed = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        prior = pd.read_csv(out_path)
        completed = set(prior["query_id"])
        print(f"Resuming: {len(completed)} already done in {out_path}", flush=True)
    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    mode = "w" if write_header else "a"

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for i, r in df.iterrows():
            if r["query_id"] in completed:
                continue
            t0 = time.perf_counter()
            answer = generate(r["query"], r["retrieved_context"])
            gen_s = time.perf_counter() - t0
            writer.writerow({
                "query_id": r["query_id"], "config": args.label, "query": r["query"],
                "reference_answer": r["reference_answer"], "retrieved_context": r["retrieved_context"],
                "generated_answer": answer, "generation_s": f"{gen_s:.4f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            f.flush()
            print(f"  [{i+1}/{len(df)}] {r['query_id']}: {gen_s:.2f}s", flush=True)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
