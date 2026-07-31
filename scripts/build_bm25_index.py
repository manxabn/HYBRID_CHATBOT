"""
Build a BM25 index (rank_bm25.BM25Okapi) over the same chunks in
data/corpus.jsonl that scripts/build_chroma_index.py embeds into ChromaDB.

Tokenization: lowercased \\w+ tokens, no stemming. Stemming would collapse
alphanumeric course codes like "CSE220" together with unrelated tokens and
undermine the exact-match precision that is BM25's whole justification in
this design (see paper.tex Section 3.3).

k1=1.5, b=0.4 (2026-08-01): b was previously left at rank_bm25's default
(0.75), tuned by BM25's original authors for variable-length web
documents -- never checked against this corpus. Swept a small k1/b grid
(scripts/test_bm25_k1_b_tuning.py) and verified end-to-end through the
real retriever (not just isolated BM25 scoring): b=0.4 gives a real,
non-negative improvement (recall@1 0.940->0.945 overall, open-ended
0.95->0.96, entity-heavy byte-identical since exact-match dominates
there regardless of BM25 params) -- this corpus's chunks are fairly
uniform in length (500-char windows), so less length normalization than
the web-document-tuned default fits better. Confirmed no regression on
any subset or metric before adopting.
"""

import json
import pickle
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
OUT_PATH = ROOT / "data" / "bm25_corpus.pkl"

from pipeline.tokenizer import tokenize


def main():
    doc_ids = []
    texts = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            texts.append(rec["text"])

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.4)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids, "texts": texts}, f)

    print(f"Built BM25 index over {len(doc_ids)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
