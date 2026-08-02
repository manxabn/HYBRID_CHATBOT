"""
Re-runs measure_nli_faithfulness.py's exact methodology (imported, module
-level paths overridden, not reimplemented) against the roundP
faithfulness regeneration -- see bootstrap_faithfulness_significance_
roundP.py's docstring for why. CPU-only, same as the original (no GPU/
Ollama contention risk).

Usage: python scripts/measure_nli_faithfulness_roundP.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_nli_faithfulness as nli

nli.INPUT_FILES = {
    "baselines": ROOT / "results" / "faithfulness_sample_baselines_roundP_faithfulness.csv",
    "novel": ROOT / "results" / "faithfulness_sample_novel_roundP_faithfulness.csv",
}
nli.OUT_PER_QUERY = ROOT / "results" / "nli_faithfulness_per_query_roundP.csv"
nli.OUT_SUMMARY = ROOT / "results" / "nli_faithfulness_summary_roundP.csv"
nli.OUT_AGREEMENT = ROOT / "results" / "nli_vs_llmjudge_agreement_roundP.csv"

if __name__ == "__main__":
    nli.main()
