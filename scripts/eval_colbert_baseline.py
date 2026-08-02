"""
Real external late-interaction retrieval baselines, evaluated on the exact
same 200-query test set, corpus, relevance judgments, and metrics as
scripts/measure_ir_metrics.py -- so the comparison to the deployed
adaptive hybrid pipeline is apples-to-apples, not a different benchmark
dressed up as one. is_relevant/recall_at_k/rr/ndcg_at_k/bootstrap_ci_diff
are imported directly from that script, not reimplemented, to guarantee
identical scoring logic.

Parameterized over --model/--label so the same methodology runs against
more than one checkpoint without duplicating the script:
  - NohTow/colbertv2.0 (Santhanam et al. 2022, NAACL) -- the original run.
  - lightonai/GTE-ModernColBERT-v1 (LightOn, PyLate-native, "first to beat
    ColBERT-small on BEIR" per the vendor's own 2025/2026 announcement,
    independently verified via WebFetch before use, not taken on faith) --
    a genuinely more current late-interaction checkpoint, added on request
    to check whether the deployed system's advantage holds against a
    newer external model too, not just a 2022-era one.
Both are loaded via PyLate (the maintained sentence-transformers-native
implementation both checkpoints target).

Exhaustive (brute-force) MaxSim, no PLAID/Voyager approximate index: the
corpus is only 7138 short chunks, well within reach of exact late-
interaction scoring on a single consumer GPU (RTX 3050 Ti, 4GB VRAM) in one
pass, so no approximation is introduced.

This is a genuine, user-authorized external-model comparison (2026-08-01;
an earlier attempt to load the first of these models was rejected by the
user via a tool-use denial and explicitly required renewed, separate
consent before retrying -- see CLAUDE.md.md). Whatever this script finds
-- either baseline winning, losing, or a mixed result -- is reported as-is
in the paper; nothing here is discarded, reworded, or rerun-until-favorable
based on which system comes out ahead.

Usage:
  python scripts/eval_colbert_baseline.py --model NohTow/colbertv2.0 --label colbert_external
  python scripts/eval_colbert_baseline.py --model lightonai/GTE-ModernColBERT-v1 --label gte_moderncolbert
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pylate.models import ColBERT
from pylate.scores import colbert_scores

from scripts.measure_ir_metrics import (
    is_relevant, recall_at_k, rr, ndcg_at_k, bootstrap_ci_diff, TOP_K, N_BOOTSTRAP, SEED,
)

CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
IR_METRICS_PATH = ROOT / "results" / "ir_metrics.csv"
DOC_BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_corpus():
    doc_ids, texts, metadatas = [], [], []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            texts.append(rec["text"])
            metadatas.append(rec["metadata"])
    return doc_ids, texts, metadatas


def pad_batch(embeddings):
    """embeddings: list of (seq_len_i, dim) tensors -> (batch, max_len, dim) + mask (batch, max_len)."""
    max_len = max(e.shape[0] for e in embeddings)
    dim = embeddings[0].shape[1]
    batch = torch.zeros(len(embeddings), max_len, dim)
    mask = torch.zeros(len(embeddings), max_len)
    for i, e in enumerate(embeddings):
        batch[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = 1.0
    return batch, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NohTow/colbertv2.0")
    parser.add_argument("--label", default="colbert_external")
    args = parser.parse_args()

    out_metrics = ROOT / "results" / f"{args.label}_baseline_raw.csv"
    out_sig = ROOT / "results" / f"{args.label}_baseline_vs_adaptive_significance.csv"

    doc_ids, texts, metadatas = load_corpus()
    print(f"Model: {args.model} | label={args.label} | Corpus: {len(doc_ids)} docs | device={DEVICE}", flush=True)

    model = ColBERT(model_name_or_path=args.model, device=DEVICE)

    print("Encoding corpus (document mode)...", flush=True)
    doc_embeddings = model.encode(
        texts, batch_size=64, is_query=False, show_progress_bar=True, convert_to_numpy=True,
    )
    doc_embeddings = [torch.as_tensor(e, dtype=torch.float32) for e in doc_embeddings]

    df = pd.read_csv(QUERIES_PATH)
    print(f"Encoding {len(df)} queries...", flush=True)
    query_embeddings = model.encode(
        df["query"].tolist(), batch_size=64, is_query=True, show_progress_bar=True, convert_to_numpy=True,
    )
    # Not all late-interaction models pad queries to a fixed length the way
    # the original ColBERT query-augmentation convention does (colbertv2.0
    # returns fixed length-32 per query; GTE-ModernColBERT-v1 does not, and
    # torch.stack on ragged lengths crashes) -- pad+mask explicitly rather
    # than assume a fixed shape, same treatment already used for documents.
    query_tensors = [torch.as_tensor(e, dtype=torch.float32) for e in query_embeddings]
    query_batch, query_mask = pad_batch(query_tensors)
    query_batch, query_mask = query_batch.to(DEVICE), query_mask.to(DEVICE)
    n_queries = query_batch.shape[0]

    all_scores = torch.zeros(n_queries, len(doc_embeddings))
    for start in range(0, len(doc_embeddings), DOC_BATCH):
        chunk = doc_embeddings[start:start + DOC_BATCH]
        doc_batch, doc_mask = pad_batch(chunk)
        doc_batch, doc_mask = doc_batch.to(DEVICE), doc_mask.to(DEVICE)
        with torch.no_grad():
            scores = colbert_scores(query_batch, doc_batch, queries_mask=query_mask, documents_mask=doc_mask)  # (n_queries, len(chunk))
        all_scores[:, start:start + DOC_BATCH] = scores.detach().cpu()
        print(f"  scored docs [{start}:{start + len(chunk)}]", flush=True)

    rows = []
    for qi, (_, r) in enumerate(df.iterrows()):
        top_idx = torch.argsort(all_scores[qi], descending=True)[:TOP_K].tolist()
        results = [{"doc_id": doc_ids[i], "text": texts[i], "metadata": metadatas[i]} for i in top_idx]
        rel_ranks = [i for i, c in enumerate(results[:TOP_K])
                     if is_relevant(c, r["query"], str(r["reference_answer"]), r["is_entity_heavy"])]
        rows.append({
            "query_id": r["query_id"], "config": args.label, "is_entity_heavy": r["is_entity_heavy"],
            "recall@1": recall_at_k(rel_ranks, 1), "recall@3": recall_at_k(rel_ranks, 3),
            "recall@5": recall_at_k(rel_ranks, 5), "mrr": rr(rel_ranks),
            "ndcg@5": ndcg_at_k(rel_ranks, 5), "ndcg@10": ndcg_at_k(rel_ranks, TOP_K),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_metrics, index=False)

    metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]
    print(f"\n=== {args.label} baseline (n={len(result_df)}) ===")
    print(result_df[metrics].mean().round(4))

    ir = pd.read_csv(IR_METRICS_PATH)
    adaptive = ir[ir["config"] == "adaptive"].set_index("query_id")
    result_indexed = result_df.set_index("query_id")

    sig_rows = []
    for metric in metrics:
        a = adaptive.loc[result_indexed.index, metric].values
        b = result_indexed[metric].values
        mean_diff, lo, hi, p_approx = bootstrap_ci_diff(a, b, n_boot=N_BOOTSTRAP, seed=SEED)
        sig_rows.append({
            "comparison": f"adaptive_vs_{args.label}", "metric": metric,
            "adaptive_mean": round(float(np.mean(a)), 4), f"{args.label}_mean": round(float(np.mean(b)), 4),
            "mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "p_approx": round(p_approx, 4), "significant": not (lo <= 0 <= hi),
        })
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out_sig, index=False)
    print(f"\n=== adaptive vs. {args.label} (paired bootstrap, 2000 resamples) ===")
    print(sig_df.to_string(index=False))
    print(f"\nWrote {out_metrics} and {out_sig}")


if __name__ == "__main__":
    main()
