"""
Quick inspection tool for the persistent ChromaDB index (chroma_db/).

Prints every collection's name and chunk count, then a bounded sample of
documents (text truncated, embeddings summarized by dimension only -- the
old version dumped every full embedding vector for every chunk, which at
7,000+ chunks x 384 floats was unreadable and effectively useless).

Usage:
    python inspect_chroma.py                 # first collection, 5 sample docs
    python inspect_chroma.py --collection ablation_corpus --n 10
"""

import argparse
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"  # anchored to the repo, not the caller's CWD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None,
                        help="Collection to sample (default: the first one listed)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of sample documents to print (default 5)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collections = client.list_collections()

    print(f"ChromaDB at {CHROMA_DIR}")
    print("Available collections:")
    for name in collections:
        count = client.get_collection(name).count()
        print(f"  {name}: {count} chunks")

    if not collections:
        print("No collections found.")
        return

    target = args.collection or collections[0]
    collection = client.get_collection(target)
    results = collection.get(limit=args.n, include=["embeddings", "documents", "metadatas"])

    print(f"\nFirst {len(results['ids'])} documents in '{target}':")
    for i, doc in enumerate(results["documents"]):
        embedding = results["embeddings"][i]
        text = doc if len(doc) <= 200 else doc[:200] + "..."
        print(f"\n[{results['ids'][i]}]")
        print(f"  Text: {text!r}")
        print(f"  Metadata: {results['metadatas'][i]}")
        print(f"  Embedding: {len(embedding)}-dim vector "
              f"(first 3: {[round(float(x), 4) for x in embedding[:3]]})")


if __name__ == "__main__":
    main()
