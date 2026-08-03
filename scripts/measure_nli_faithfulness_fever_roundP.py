"""
A further, still genuinely untried angle on the independent NLI
faithfulness cross-check's weak row-level agreement with the LLM judge,
after the first model swap (measure_nli_faithfulness_deberta_roundP.py,
same architecture size, generic MNLI-only training) moved Pearson r from
-0.300 to -0.076 but not into positive territory.

This time the change is not just "bigger," it is "trained on a task
closer to what we're actually asking": MoritzLaurer/DeBERTa-v3-large-mnli
-fever-anli-ling-wanli is fine-tuned on MultiNLI + Fever-NLI + ANLI +
LingNLI + WANLI (885k pairs total). Fever-NLI in particular reformulates
FEVER (Thorne et al. 2018, "FEVER: a Large-scale Dataset for Fact
Extraction and VERification") as NLI pairs -- claim vs. retrieved
Wikipedia evidence, i.e. exactly the "does this evidence support this
claim" structure our SummaC-ZS per-sentence check already uses, not
generic human-written entailment pairs (SNLI/MultiNLI alone, what both
prior models were trained on). A model that has specifically seen
claim-vs-evidence verification data is a better-motivated candidate for
this task than a same-size model that has only seen entailment data,
independent of raw parameter count.

Verified this loads through the SAME sentence_transformers.CrossEncoder
interface already used elsewhere in this project (no trust_remote_code,
no extra pip package -- deliberately avoided two other candidates that
would have needed either, since introducing a new remote-code-execution
or arbitrary-git-install trust surface into the project for a measurement
-only experiment was judged not worth it without checking with the user
first). Its label ordering was independently confirmed against its own
config (id2label = {0: entailment, 1: neutral, 2: contradiction}) --
DIFFERENT from the previous two models' {0: contradiction, 1: entailment,
2: neutral} -- so LABELS below is deliberately NOT the same list reused
by measure_nli_faithfulness.py/measure_nli_faithfulness_deberta_roundP.py;
using the wrong order here would have silently scored contradiction as
entailment.

Runs the exact same SummaC-ZS methodology (imported, only NLI_MODEL/LABELS
and I/O paths overridden) against the identical 169-row roundP sample used
for both prior measurements, for a direct three-way comparison. Pure
evaluation-instrument experiment -- no pipeline/generation/retrieval code
touched.

GPU run (2026-08-03, after the CPU attempt failed): the first attempt ran
on CPU (matching every other NLI script in this project) and hit a
genuine performance anomaly -- 9+ hours, only 8/38 batches completed, the
first batch alone took 3h08m -- not a normal "bigger model, proportionally
slower" curve, something pathological about this specific model class on
CPU-only inference on this hardware. Verified before switching: `tasklist`
confirmed no `llama-server.exe` (Ollama's GPU-using child process) running,
so there is no concurrent GPU job to contend with; `torch.cuda.is_available()`
confirmed the RTX 3050 Ti Laptop GPU is visible with ~3.45GB free VRAM.
Uses a reduced batch size (16, vs.\ the default 64) to stay safely within
that limited VRAM budget for a ~435M-param model.

Usage: python scripts/measure_nli_faithfulness_fever_roundP.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_nli_faithfulness as nli

nli.NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
nli.LABELS = ["entailment", "neutral", "contradiction"]  # confirmed against this model's own config.id2label -- see module docstring
nli.DEVICE = "cuda"
nli.BATCH_SIZE = 16
nli.INPUT_FILES = {
    "baselines": ROOT / "results" / "faithfulness_sample_baselines_roundP_faithfulness.csv",
    "novel": ROOT / "results" / "faithfulness_sample_novel_roundP_faithfulness.csv",
}
nli.OUT_PER_QUERY = ROOT / "results" / "nli_faithfulness_per_query_roundP_fever.csv"
nli.OUT_SUMMARY = ROOT / "results" / "nli_faithfulness_summary_roundP_fever.csv"
nli.OUT_AGREEMENT = ROOT / "results" / "nli_vs_llmjudge_agreement_roundP_fever.csv"

if __name__ == "__main__":
    nli.main()
