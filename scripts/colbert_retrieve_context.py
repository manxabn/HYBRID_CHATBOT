"""
Step 1 of the ColBERT-as-retriever end-to-end generation-quality ablation
(companion to scripts/colbert_generate_and_score.py, step 2): retrieves
each of the 200 test queries' top-5 context under a given late-interaction
model, exactly like scripts/run_ablation.py's retrieval step for the
existing BM25-only/Vector-only/No-retrieval rows in Table~\ref{tab:ablation}
(same "\n\n".join(top-5 chunk text) construction), so the two are directly
comparable.

Runs in the isolated .venv_colbert (needs pylate/CUDA torch); the actual
LLM generation + BLEU/ROUGE-L/BERTScore/METEOR scoring happens in step 2,
which runs in the main project venv instead (Ollama HTTP client +
bert-score/rouge-score/nltk all live there, not in .venv_colbert).

Usage:
  python scripts/colbert_retrieve_context.py --model NohTow/colbertv2.0 --label colbert_external
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pylate.models import ColBERT
from pylate.scores import colbert_scores

CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
TOP_K = 5
DOC_BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_corpus():
    doc_ids, texts = [], []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            texts.append(rec["text"])
    return doc_ids, texts


def pad_batch(embeddings):
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

    out_path = ROOT / "results" / f"{args.label}_retrieved_context.csv"

    doc_ids, texts = load_corpus()
    print(f"Model: {args.model} | Corpus: {len(doc_ids)} docs | device={DEVICE}", flush=True)
    model = ColBERT(model_name_or_path=args.model, device=DEVICE)

    doc_embeddings = model.encode(texts, batch_size=64, is_query=False, show_progress_bar=True, convert_to_numpy=True)
    doc_embeddings = [torch.as_tensor(e, dtype=torch.float32) for e in doc_embeddings]

    df = pd.read_csv(QUERIES_PATH)
    query_embeddings = model.encode(df["query"].tolist(), batch_size=64, is_query=True, show_progress_bar=True, convert_to_numpy=True)
    # Pad+mask explicitly -- not every late-interaction model pads queries to
    # a fixed length (colbertv2.0 does; GTE-ModernColBERT-v1 does not).
    query_tensors = [torch.as_tensor(e, dtype=torch.float32) for e in query_embeddings]
    query_batch, query_mask = pad_batch(query_tensors)
    query_batch, query_mask = query_batch.to(DEVICE), query_mask.to(DEVICE)

    all_scores = torch.zeros(query_batch.shape[0], len(doc_embeddings))
    for start in range(0, len(doc_embeddings), DOC_BATCH):
        chunk = doc_embeddings[start:start + DOC_BATCH]
        doc_batch, doc_mask = pad_batch(chunk)
        doc_batch, doc_mask = doc_batch.to(DEVICE), doc_mask.to(DEVICE)
        with torch.no_grad():
            scores = colbert_scores(query_batch, doc_batch, queries_mask=query_mask, documents_mask=doc_mask)
        all_scores[:, start:start + DOC_BATCH] = scores.detach().cpu()
        print(f"  scored docs [{start}:{start + len(chunk)}]", flush=True)

    rows = []
    for qi, (_, r) in enumerate(df.iterrows()):
        top_idx = torch.argsort(all_scores[qi], descending=True)[:TOP_K].tolist()
        context = "\n\n".join(texts[i] for i in top_idx)
        rows.append({
            "query_id": r["query_id"], "query": r["query"], "reference_answer": r["reference_answer"],
            "is_entity_heavy": r["is_entity_heavy"], "retrieved_context": context,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
