"""
Build a BM25 index (rank_bm25.BM25Okapi) over the same chunks in
data/corpus.jsonl that scripts/build_chroma_index.py embeds into ChromaDB.

Tokenization: lowercased \\w+ tokens, no stemming. Stemming would collapse
alphanumeric course codes like "CSE220" together with unrelated tokens and
undermine the exact-match precision that is BM25's whole justification in
this design (see paper.tex Section 3.3).
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
    bm25 = BM25Okapi(tokenized)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids, "texts": texts}, f)

    print(f"Built BM25 index over {len(doc_ids)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
