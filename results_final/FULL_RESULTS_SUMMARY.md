# Full Results Summary

Consolidated view of every real, script-verified result referenced in the paper, gathered here from across `results/` (the original archive, untouched) into one place. Nothing here is a new measurement unless explicitly marked **(new)** — this file cross-references, it does not re-derive.

## 1. Retrieval Quality (`retrieval/`)

**Recall@k / MRR / nDCG@k** — `retrieval/ir_metrics.csv`, `retrieval/ir_metrics_bootstrap_significance.csv` (n=200, English test set, paired bootstrap significance, adaptive vs. each baseline)

| Config | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| adaptive | tied with full_hybrid/bm25_only on all metrics (p≥0.530); see paper Section "Novel vs. Baselines" for full table |

**Precision@k / MAP (new)** — `retrieval/precision_map.csv`

| Config | P@1 | P@3 | P@5 | P@10 | MAP |
|---|---|---|---|---|---|
| adaptive | 0.985 | 0.733 | 0.562 | 0.376 | 0.961 |
| full_hybrid | 0.985 | 0.730 | 0.564 | 0.376 | 0.960 |
| bm25_only | 0.975 | 0.715 | 0.551 | 0.371 | 0.952 |
| vector_only | 0.965 | 0.660 | 0.501 | 0.315 | 0.897 |

Same relevance judgment as Recall/MRR/nDCG — confirms the same parity ordering, not a new finding.

**Fusion**: linear (default), `Score = λ·S_bm25 + (1-λ)·S_vec`; adaptive routing λ_entity=0.9, λ_open=0.5. RRF (k=60) implemented, not the deployed default.

## 2. Faithfulness / Generation (`faithfulness/`)

**LLM-judge faithfulness** (RAGAS-style) — adaptive_novel 0.861, highest of 4 configs; matched-subset paired test: significant only vs. no-retrieval (+0.236, p=0.007), non-significant tie vs. BM25-only/full-hybrid/vector-only.

**BLEU/ROUGE-L/BERTScore/METEOR** — full parity vs. BM25-only and full-hybrid on all four (p≥0.530 both tests); vs. vector-only, METEOR test-disagreement (p=0.128 t, p=0.018 Wilcoxon) → inconclusive by this project's own convention.

**NLI faithfulness cross-check** — `faithfulness/nli_vs_llmjudge_agreement_roundP*.csv`, three models on the identical 169-row sample:

| Model | Pearson r | Binary agreement |
|---|---|---|
| MiniLM (original) | -0.300 | 0.556 |
| DeBERTa-base | -0.076 | 0.822 |
| DeBERTa-large-FEVER (best) | -0.018 | 0.905 |

Root-cause quantified (not just measured): ceiling effect, NLI's structural blind spot on hedge/decline answers, structured-record answers, and one confirmed LLM-judge scoring error — see paper Limitations "Seventh" for the full breakdown.

**MiniCheck (4th model)**: attempted, blocked by a real environment/security constraint (CVE-2025-32434, no safetensors checkpoint available), not forced through.

## 3. Abstention Gate (`ablation/`)

| Stage | CV accuracy |
|---|---|
| Single threshold (original) | 0.612 |
| 4-signal logistic regression | 0.644 |
| **Gradient-boosted trees (deployed)** | **0.737** |

Both improvements validated across 5 independent fold-assignment seeds; the GBT result additionally verified for overfitting (in-sample 0.873 vs. CV 0.737, a normal gap) and stability across 5 classifier random seeds (0.7374–0.7390). Entity-heavy route unchanged (0.967 CV accuracy, single threshold).

## 4. Reranker (`ablation/`)

Every configuration tested tops out at a tie, never a win: generic (loss), fine-tuned (loss), pool-restricted (loss shrinks to noise), route-conditional (clean tie vs. reranker-off; ties BM25-only/vector-only vs. external baselines, borderline/inconclusive vs. full-hybrid on 2/4 metrics), route-conditional + pool-restricted combined (null result, adds nothing beyond route-conditional alone). Deployed default: reranker off.

## 5. Efficiency (`efficiency/`)

**Latency percentiles** — `efficiency/latency_percentiles.csv`: retrieval p50 ~20ms, p99 ~35ms across all configs (negligible). Generation dominates: p50 5.1–5.3s, p99 17–19s (retrieval-augmented configs); no-retrieval baseline p50 9.1s (longer, less-grounded responses).

## 6. Robustness (`robustness/`)

**Consistency (new)** — `robustness/consistency_check.csv`, 20 real queries × 3 repeats: retrieval 100% consistent (20/20); generation 85% byte-identical (17/20) despite temperature=0/fixed seed, mean similarity 0.970 on the non-identical cases — direct confirmation of the previously-anecdotal GPU floating-point non-determinism.

## What's deliberately not here

- **BEIR / HotpotQA / MS MARCO / scalability curves to 1M docs**: out of scope for this system (a fixed, curated, single-institution bilingual corpus, not a general-purpose web-scale retriever) — including them would evaluate a different paper's claims, not this one's.
- **Human evaluation**: explicitly excluded per instruction; machine-side prep (a 50-item disagreement-prioritized annotation sample) exists in `results/` but the annotation itself requires a person.
- **Multi-seed reruns of the main generation-quality numbers**: the calibration work (abstention, NLI models) already uses 5 seeds; the main BLEU/ROUGE/faithfulness numbers are single-run, a real, disclosed limitation, not silently omitted.

## Source of truth

Every number above traces to a script under `scripts/` and an output file under `results/` or `results_final/`. `results/` is the original, untouched archive; `results_final/` is this consolidation plus the three genuinely new measurements (Precision@k/MAP, latency percentiles, consistency). Nothing was deleted or overwritten in either location.
