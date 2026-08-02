"""
A genuinely untried angle on the independent NLI faithfulness cross-check's
persistently weak (and, on the last two re-measurements, chance-level or
worse) row-level agreement with the LLM judge (Pearson r: 0.089 -> -0.063
-> -0.300 across three rounds, paper.tex Section subsec:nli-faithfulness /
Limitations "Seventh"). Every prior re-measurement kept the same NLI model
(cross-encoder/nli-MiniLM2-L6-H768, chosen originally for RAM reasons while
a concurrent Ollama job was using most of available memory, see measure_nli
_faithfulness.py's module docstring) and only re-ran it against fresh data
-- never actually asked whether a stronger NLI model changes the picture.

This re-runs the EXACT same SummaC-ZS methodology (imported from measure_
nli_faithfulness.py, not reimplemented) against the same roundP faithfulness
sample already used for the deployed r=-0.300 measurement, swapping only
the model: cross-encoder/nli-deberta-v3-base (~370M params, ~4x the
MiniLM variant's ~66M) -- a real, standard, more capable pretrained NLI
model, same label ordering confirmed against its model card (contradiction/
entailment/neutral). RAM is no longer the constraint it was when the
smaller model was chosen: no Ollama-bound job is running concurrently in
this session (`tasklist` confirms ollama.exe idle, no llama-server.exe
child), and ~7.7GB is free of ~16GB total -- comfortably enough headroom
for a ~1.4GB fp32 model doing CPU inference on ~200 rows.

This is a pure EVALUATION-INSTRUMENT experiment: it does not touch
pipeline/conformal_abstention.py (which still uses the MiniLM model for
production claim-level backoff) or any generation/retrieval behavior --
only which model is used to independently SCORE already-generated answers
after the fact. Safe to run freely; if it doesn't help, that is reported
honestly exactly like every other measurement in this project.

Usage: python scripts/measure_nli_faithfulness_deberta_roundP.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_nli_faithfulness as nli

nli.NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
nli.INPUT_FILES = {
    "baselines": ROOT / "results" / "faithfulness_sample_baselines_roundP_faithfulness.csv",
    "novel": ROOT / "results" / "faithfulness_sample_novel_roundP_faithfulness.csv",
}
nli.OUT_PER_QUERY = ROOT / "results" / "nli_faithfulness_per_query_roundP_deberta.csv"
nli.OUT_SUMMARY = ROOT / "results" / "nli_faithfulness_summary_roundP_deberta.csv"
nli.OUT_AGREEMENT = ROOT / "results" / "nli_vs_llmjudge_agreement_roundP_deberta.csv"

if __name__ == "__main__":
    nli.main()
