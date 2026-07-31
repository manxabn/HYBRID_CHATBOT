"""
Builds a FRESH ChromaDB collection into a staging directory (chroma_db_
staging/), not the live chroma_db/ -- lets the expensive re-embedding step
(re-embedding all 7050 corpus chunks with the newly-deployed Banglish-
expanded model) run WITHOUT needing to touch or lock the live directory,
which other still-running processes (retrieval-dependent ablations) have
open. Once those finish and release their file handles, swap chroma_db_
staging/ into chroma_db/ (a near-instant directory rename, not a
re-embed) via scripts/swap_chroma_staging.py.

Same embedding logic as scripts/build_chroma_index.py, just a different
target directory -- not a new rebuild method.

Usage: python scripts/stage_chroma_index_rebuild.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from pipeline.chroma_embedding import Chroma1xEmbeddingFunction

CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
STAGING_DIR = ROOT / "chroma_db_staging"
COLLECTION_NAME = "ablation_corpus"


def main():
    if STAGING_DIR.exists():
        print(f"Removing old staging dir {STAGING_DIR}")
        shutil.rmtree(STAGING_DIR)

    client = chromadb.PersistentClient(path=str(STAGING_DIR))
    embedding_function = Chroma1xEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    doc_ids, texts, metadatas = [], [], []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            texts.append(rec["text"])
            metadatas.append(rec["metadata"])

    batch_size = 100
    for i in range(0, len(doc_ids), batch_size):
        collection.add(
            ids=doc_ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        print(f"Embedded {min(i + batch_size, len(doc_ids))}/{len(doc_ids)}")

    print(f"Staged Chroma collection '{COLLECTION_NAME}' now has {collection.count()} chunks "
          f"at {STAGING_DIR} (NOT yet live -- run scripts/swap_chroma_staging.py to activate)")


if __name__ == "__main__":
    main()
