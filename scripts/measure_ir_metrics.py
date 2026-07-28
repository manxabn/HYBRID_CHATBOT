"""
Standard IR retrieval-quality metrics (Recall@k, MRR, nDCG@10), the
"latest measurement meters" gap identified directly by the user: this
project's evaluation so far has been end-to-end generation-quality metrics
(BLEU/ROUGE-L/BERTScore/METEOR) plus a few one-off retrieval accuracy
checks (lambda sweeps, the unambiguous-match test), but never a proper
Recall@k/MRR/nDCG table with paired significance testing across retrieval
configurations -- the standard reporting format in 2025-2026 hybrid-
retrieval benchmarking literature.

Retrieval-only (no LLM generation, no Ollama) -- safe to run alongside any
concurrently-running Ollama-bound job without competing for that resource.

Relevance judgment (binary, per query):
  - Open-ended queries: reference_answer is the corpus's own verbatim
    EnglishQA Answer field (paper.tex Section 4.2: "100 open-ended queries
    sampled directly from its natural-language question/answer pairs") --
    a candidate is relevant iff reference_answer is a substring of its text.
  - Entity-heavy queries: reference_answer is a COMPOSED sentence, not a
    verbatim chunk (paper.tex: "reference answer composed directly from the
    relevant structured fields"), so substring match doesn't apply --
    reused directly from scripts/test_unambiguous_match.py's _is_correct
    logic (course-code / email / faculty-initial-name entity matching).

Configurations compared: bm25_only (lambda=1, linear), vector_only
(lambda=0, linear), full_hybrid (lambda=0.5, linear), and adaptive (the
production retrieve_adaptive routing).

Usage: python scripts/measure_ir_metrics.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.hybrid_retriever as hr
from pipeline.hybrid_retriever import COURSE_CODE_RE, FACULTY_INITIAL_RE, _normalize_name

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "ir_metrics.csv"
OUT_BOOTSTRAP = ROOT / "results" / "ir_metrics_bootstrap_significance.csv"
TOP_K = 10
N_BOOTSTRAP = 2000
SEED = 42


def is_relevant(candidate, query: str, reference_answer: str, is_entity_heavy: bool) -> bool:
    if not is_entity_heavy:
        return str(reference_answer).strip() in candidate["text"]
    ref_codes = {m.upper() for m in COURSE_CODE_RE.findall(reference_answer)}
    ref_emails = {m.lower() for m in EMAIL_RE.findall(reference_answer)}
    text_upper = candidate["text"].upper()
    text_lower = candidate["text"].lower()
    if ref_codes:
        return all(code in text_upper for code in ref_codes)
    if ref_emails:
        return all(email in text_lower for email in ref_emails)
    query_codes = {m.upper() for m in COURSE_CODE_RE.findall(query)}
    if query_codes:
        cand_course = str(candidate["metadata"].get("Course", "")).upper()
        cand_canonical = COURSE_CODE_RE.match(cand_course)
        cand_canonical = cand_canonical.group(0) if cand_canonical else ""
        return cand_canonical in query_codes
    cand_initial = str(candidate["metadata"].get("Initial", "")).upper()
    if cand_initial and any(m.group(0) == cand_initial for m in FACULTY_INITIAL_RE.finditer(query.upper())):
        return True
    cand_name = _normalize_name(str(candidate["metadata"].get("Name", "")))
    norm_query = _normalize_name(query)
    return bool(cand_name) and cand_name in norm_query


def recall_at_k(rel_ranks, k):
    return 1.0 if any(r < k for r in rel_ranks) else 0.0


def rr(rel_ranks):
    return 1.0 / (min(rel_ranks) + 1) if rel_ranks else 0.0


def ndcg_at_k(rel_ranks, k):
    """IDCG must be normalized by however many relevant docs actually exist
    in the returned candidate list (R = len(rel_ranks)), not by a fixed
    assumption of exactly one relevant doc. This corpus has substantial
    near-duplicate/paraphrase content (multiple chunks can genuinely be
    relevant to the same query), so R > 1 is common, not an edge case --
    assuming R=1 was a real bug in an earlier version of this function,
    caught by nDCG@10 values exceeding 1.0 (impossible for a correctly
    normalized metric) on the first run."""
    dcg = sum(1.0 / np.log2(r + 2) for r in rel_ranks if r < k)
    n_relevant = min(len(rel_ranks), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def run_config(retriever, df, config_fn, label):
    rows = []
    for _, r in df.iterrows():
        results = config_fn(retriever, r["query"])
        rel_ranks = [i for i, c in enumerate(results[:TOP_K])
                     if is_relevant(c, r["query"], str(r["reference_answer"]), r["is_entity_heavy"])]
        rows.append({
            "query_id": r["query_id"], "config": label, "is_entity_heavy": r["is_entity_heavy"],
            "recall@1": recall_at_k(rel_ranks, 1), "recall@3": recall_at_k(rel_ranks, 3),
            "recall@5": recall_at_k(rel_ranks, 5), "mrr": rr(rel_ranks),
            # nDCG@5 added as the headline metric alongside nDCG@10, per
            # SemEval-2026 Task 8 (MTRAGEval)'s reporting convention: nDCG@5
            # primary, nDCG@10/Recall@10 secondary -- confirmed via direct
            # search of the shared-task's own system-description papers
            # (e.g. AILS-NTUA at SemEval-2026 Task 8), not assumed.
            "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
        })
    return pd.DataFrame(rows)


def bootstrap_ci_diff(a, b, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_approx = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), lo, hi, p_approx


def main():
    df = pd.read_csv(QUERIES_PATH)
    retriever = hr.HybridRetriever()

    configs = {
        "bm25_only": lambda r, q: r.retrieve(q, 1.0, fusion="linear", top_n=TOP_K),
        "vector_only": lambda r, q: r.retrieve(q, 0.0, fusion="linear", top_n=TOP_K),
        "full_hybrid": lambda r, q: r.retrieve(q, 0.5, fusion="linear", top_n=TOP_K),
        "adaptive": lambda r, q: r.retrieve_adaptive(q, top_n=TOP_K)[0],
    }

    all_results = {}
    for label, fn in configs.items():
        print(f"Running {label}...", flush=True)
        all_results[label] = run_config(retriever, df, fn, label)

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)

    summary = combined.groupby("config")[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4)
    print("\n=== Overall (n=200) ===")
    print(summary)

    print("\n=== By query type ===")
    by_type = combined.groupby(["config", "is_entity_heavy"])[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4)
    print(by_type)

    # Paired bootstrap significance: adaptive vs each baseline, per metric.
    bootstrap_rows = []
    metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]
    adaptive = all_results["adaptive"].set_index("query_id")
    for other_label in ["bm25_only", "vector_only", "full_hybrid"]:
        other = all_results[other_label].set_index("query_id")
        for metric in metrics:
            a = adaptive.loc[other.index, metric].values
            b = other[metric].values
            mean_diff, lo, hi, p_approx = bootstrap_ci_diff(a, b)
            bootstrap_rows.append({
                "comparison": f"adaptive_vs_{other_label}", "metric": metric,
                "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "p_approx": round(p_approx, 4), "significant": not (lo <= 0 <= hi),
            })
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(OUT_BOOTSTRAP, index=False)
    print("\n=== Paired bootstrap significance (adaptive vs. baselines, 2000 resamples) ===")
    print(bootstrap_df.to_string(index=False))
    print(f"\nWrote {OUT_PATH} and {OUT_BOOTSTRAP}")


if __name__ == "__main__":
    main()
