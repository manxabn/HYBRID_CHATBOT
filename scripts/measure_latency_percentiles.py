"""
Latency percentiles (p50/p95/p99), not just means -- a genuine, previously
-missing measurement identified against an external "A* evaluation
framework" checklist. Computed entirely from timing data already present
in existing raw-output CSVs (retrieval_s/generation_s/total_s logged per
query since early in this project) -- no new generation runs needed, this
is pure re-aggregation of real, already-measured per-query timings.

Isolates retrieval latency from generation latency per the checklist's
own request ("Retrieval latency... isolates retrieval cost from
generation cost"), across all five configs this project evaluates
(adaptive_novel, full_hybrid, bm25_only, vector_only, no_retrieval).

Usage: python scripts/measure_latency_percentiles.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "results_final" / "efficiency" / "latency_percentiles.csv"

SOURCES = {
    "adaptive_novel": ROOT / "results" / "novel_pipeline_raw_outputs_roundO_noreranker.csv",
    "full_hybrid": ROOT / "results" / "ablation_raw_outputs.csv",
    "bm25_only": ROOT / "results" / "ablation_raw_outputs.csv",
    "vector_only": ROOT / "results" / "ablation_raw_outputs.csv",
    "no_retrieval": ROOT / "results" / "ablation_raw_outputs.csv",
}


def percentiles(series):
    return {
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "n": int(len(series)),
    }


def main():
    rows = []
    cache = {}
    for config, path in SOURCES.items():
        if path not in cache:
            cache[path] = pd.read_csv(path)
        df = cache[path]
        subset = df[df["config"] == config] if "config" in df.columns else df
        if len(subset) == 0:
            print(f"SKIP {config}: no rows found in {path}")
            continue
        for metric in ["retrieval_s", "generation_s", "total_s"]:
            if metric not in subset.columns:
                continue
            stats = percentiles(subset[metric])
            rows.append({"config": config, "metric": metric, **stats})

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
