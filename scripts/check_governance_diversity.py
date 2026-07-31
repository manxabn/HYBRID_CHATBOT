"""
Diversity check on data/governance_augmentation_proposed.csv (scripts/
augment_governance_category.py's output): the generation script rejects a
candidate paraphrase if its cosine similarity to any ALREADY-ACCEPTED
phrasing exceeds 0.90 (checked incrementally, one candidate at a time,
against the fine-tuned embedding model). This is a different, and in
practice looser, check than clustering the WHOLE group of questions for a
fact together -- an incremental pairwise filter can still accept several
candidates that are each individually "different enough" from the ones
immediately before them, while collectively forming a tight cluster (e.g.
A,B accepted for differing from each other; C differs from A and B but
lands near the midpoint; results in 3 points that still cluster together
under whole-group agglomerative clustering, which this pairwise check alone
cannot see).

Found by direct testing (2026-07-29) that this DOES matter here, not just
in principle: whole-group agglomerative clustering (distance_threshold=0.15,
average linkage, on the base -- not fine-tuned -- MiniLM model, a
deliberately different embedding model than generation used, so this is a
genuinely independent check) finds 10 of 22 facts have their 5 questions
(4 generated + 1 source) collapse into only 1-2 clusters, while the other
12 facts show good diversity (up to 5 distinct clusters). Writes a
per-fact diversity report so this is a disclosed, traceable finding
attached to the proposed augmentation file, not just a chat statement.

Usage: python scripts/check_governance_diversity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IN_PATH = ROOT / "data" / "governance_augmentation_proposed.csv"
OUT_PATH = ROOT / "data" / "governance_augmentation_diversity_check.csv"
DISTANCE_THRESHOLD = 0.15


def main():
    # Explicitly CPU: safe to run alongside any concurrently-running
    # GPU-bound job (e.g. an embedding fine-tune) without contention risk.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    df = pd.read_csv(IN_PATH)

    rows = []
    for answer, group in df.groupby("answer"):
        questions = list(group["question"]) + [group["source_question"].iloc[0]]
        embs = model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
        sims = embs @ embs.T
        dist = np.clip(1 - sims, 0, None)
        np.fill_diagonal(dist, 0)
        n = len(questions)
        if n < 3:
            n_clusters = n
        else:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=DISTANCE_THRESHOLD,
                                                  metric="precomputed", linkage="average")
            n_clusters = len(set(clustering.fit_predict(dist)))
        rows.append({
            "answer": answer, "n_questions": n, "n_clusters": n_clusters,
            "possible_collapse": n_clusters <= n // 2,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Facts checked: {len(out)}")
    print(f"Facts flagged as possibly collapsed (clusters <= half the question count): "
          f"{out['possible_collapse'].sum()}/{len(out)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
