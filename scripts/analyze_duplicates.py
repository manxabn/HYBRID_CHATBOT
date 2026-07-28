"""
Quantify near-duplicate paraphrase rows in EnglishQA/BanglishQA.

Motivation: the novel pipeline's abstention gate misfires specifically when
several near-duplicate paraphrase chunks retrieve as top candidates (see
pipeline/hybrid_retriever.py's retrieve() docstring) -- this script measures
how common that actually is in the corpus, rather than leaving it as an
anecdotal observation from one example (Q005, "Are journals, magazines and
reference books available for borrowing?"). Uses the same embedding model
the retriever itself uses (all-MiniLM-L6-v2), NOT chromadb's persistent
client/index, so this can safely run concurrently with anything using
chroma_db (no file-lock contention -- confirmed necessary after an earlier
deadlock this session from two chroma_db-touching processes at once).

Method: embed every Question in a table, compute the full pairwise cosine
similarity matrix (cheap: ~2,300 x 384-dim vectors), and cluster rows whose
max similarity to another row exceeds DUPLICATE_THRESHOLD via union-find.
Reports cluster size distribution and writes the full cluster assignment to
results/duplicate_clusters_<table>.csv so it's independently checkable.

Usage: python scripts/analyze_duplicates.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embeddings import ChromaEmbeddingFunction

DB_PATH = ROOT / "knowledge_base.db"
DUPLICATE_THRESHOLD = 0.90
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def analyze_table(conn, table, question_col, id_col="id"):
    df = pd.read_sql(f"SELECT {id_col}, {question_col} FROM {table} WHERE {question_col} IS NOT NULL", conn)
    df = df[df[question_col].str.strip() != ""]
    print(f"\n=== {table} ({len(df)} rows) ===")

    embedder = ChromaEmbeddingFunction(MODEL_NAME)
    embeddings = np.array(embedder.embed_documents(df[question_col].tolist()))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T

    n = len(df)
    uf = UnionFind(n)
    n_pairs_above_threshold = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= DUPLICATE_THRESHOLD:
                uf.union(i, j)
                n_pairs_above_threshold += 1

    cluster_of = [uf.find(i) for i in range(n)]
    cluster_sizes = pd.Series(cluster_of).value_counts()
    multi_row_clusters = cluster_sizes[cluster_sizes > 1]

    print(f"Pairs with cosine similarity >= {DUPLICATE_THRESHOLD}: {n_pairs_above_threshold}")
    print(f"Rows belonging to a multi-row near-duplicate cluster: "
          f"{multi_row_clusters.sum()} / {n} ({100*multi_row_clusters.sum()/n:.1f}%)")
    print(f"Number of multi-row clusters: {len(multi_row_clusters)}")
    print(f"Cluster size distribution:\n{multi_row_clusters.value_counts().sort_index()}")

    df["cluster_id"] = cluster_of
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes)
    out_path = ROOT / "results" / f"duplicate_clusters_{table}.csv"
    df.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).to_csv(out_path, index=False)
    print(f"Wrote full cluster assignment to {out_path}")

    # Show a couple of concrete examples for the write-up.
    example_clusters = multi_row_clusters.index[:2]
    for cid in example_clusters:
        rows = df[df["cluster_id"] == cid][question_col].tolist()
        print(f"  Example cluster (size {len(rows)}): {rows[:4]}")

    return {"table": table, "n_rows": n, "n_pairs_above_threshold": n_pairs_above_threshold,
            "n_rows_in_multi_clusters": int(multi_row_clusters.sum()),
            "n_multi_clusters": len(multi_row_clusters)}


def main():
    conn = sqlite3.connect(DB_PATH)
    results = []
    results.append(analyze_table(conn, "EnglishQA", "Question"))
    results.append(analyze_table(conn, "BanglishQA", "QuestionBanglish"))
    conn.close()

    summary = pd.DataFrame(results)
    out_path = ROOT / "results" / "duplicate_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nWrote summary to {out_path}")
    print(summary)


if __name__ == "__main__":
    main()
