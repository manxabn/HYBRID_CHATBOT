"""
Build a fresh ChromaDB collection from data/corpus.jsonl.

The existing chroma_db/ directory is stale (built by a dead venv pointing at
a different machine, contains orphaned collection dirs, and chromadb isn't
even installed anywhere that currently runs) so it is deleted and rebuilt
from the same corpus.jsonl that scripts/build_bm25_index.py indexes, keeping
both retrieval streams over identical (doc_id, text) pairs.

WARNING (2026-07-31): this script unconditionally shutil.rmtree()s
chroma_db/ before rebuilding -- safe for the very first build (or any time
chroma_db/ is known to be unused), but NOT safe to re-run against a LIVE,
in-use index: another process (a running retriever, an evaluation job)
holding it open could hit a PermissionError mid-delete, or worse, be left
querying a collection whose files are being removed out from under it.
scripts/stage_chroma_index_rebuild.py + scripts/swap_chroma_staging.py exist
specifically to avoid this risk (build into a separate chroma_db_staging/
directory, then atomically rename into place, failing cleanly rather than
partially deleting anything if the live directory is still locked) -- use
that pair instead of this script for any rebuild after the corpus already
has a live index in production use (e.g. after the 2026-07-31 Banglish
dataset expansion, which used the staged workflow for exactly this reason).
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
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "ablation_corpus"


def main():
    if CHROMA_DIR.exists():
        print(f"Removing stale {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
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

    print(f"Chroma collection '{COLLECTION_NAME}' now has {collection.count()} chunks")


if __name__ == "__main__":
    main()
