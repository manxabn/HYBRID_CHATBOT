"""
Direct test of paraphrase robustness on general (non-entity, non-course)
EnglishQA content -- the gap the user flagged: prior test sets are heavy
on entity-heavy/course-code queries, and no dedicated check exists for
whether the retriever finds the right answer when a question is phrased
differently from how it appears in the corpus.

Uses REAL, human-authored paraphrase pairs already in the dataset (not
synthetic): EnglishQA rows sharing the same (Category, Answer) where one
is Type='Original' and another is Type='Paraphrase', restricted to
Paraphrase rows with Split in ('val','test') only -- Split='train' rows
may have been used in embedding fine-tuning, so including them would leak
training data into what should be a held-out check (same discipline this
project already applies elsewhere, e.g. build_crosslingual_stress_test_
valsplit.py).

Methodology: query with the PARAPHRASE's question text, retrieve top-5,
and check whether the ORIGINAL's own chunk (doc_id) is present -- not
merely "any chunk containing the right answer text," since the paraphrase
row itself is also separately ingested into the corpus and would trivially
match its own near-identical wording, which would not test generalization
to different phrasing at all. Reports both "found ANY correct-answer
chunk" (upper bound) and "found the ORIGINAL's specific chunk"
(what actually demonstrates paraphrase generalization).

Usage: python scripts/eval_paraphrase_robustness.py
"""

import sys
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "results" / "paraphrase_robustness_raw.csv"
TOP_K = 5


def load_pairs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT o.id, o.Category, o.Question, o.Answer, p.id, p.Question, p.Split
        FROM EnglishQA o
        JOIN EnglishQA p ON o.Category = p.Category AND o.Answer = p.Answer
                          AND o.Type='Original' AND p.Type='Paraphrase'
        WHERE p.Split IN ('val','test')
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    pairs = load_pairs()
    print(f"Held-out (Original, Paraphrase) pairs: {len(pairs)}", flush=True)

    retriever = HybridRetriever()
    rows = []
    for orig_id, category, orig_q, answer, para_id, para_q, split in pairs:
        orig_doc_id = f"EnglishQA-{orig_id}-chunk0"
        results = retriever.retrieve(para_q, 0.5, top_n=TOP_K)  # full hybrid, deployed default
        top_doc_ids = [d["doc_id"] for d in results]
        found_original_chunk = orig_doc_id in top_doc_ids
        found_any_correct = any(str(answer).strip() in d["text"] for d in results)
        rank_of_original = top_doc_ids.index(orig_doc_id) + 1 if found_original_chunk else None
        rows.append({
            "orig_id": orig_id, "para_id": para_id, "category": category, "split": split,
            "orig_question": orig_q, "para_question": para_q,
            "found_original_chunk_top5": found_original_chunk,
            "rank_of_original_chunk": rank_of_original,
            "found_any_correct_answer_top5": found_any_correct,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    n = len(out)
    n_found_orig = out["found_original_chunk_top5"].sum()
    n_found_any = out["found_any_correct_answer_top5"].sum()
    print(f"\nn={n}")
    print(f"Found ORIGINAL's own chunk in top-5 (true paraphrase generalization): {n_found_orig}/{n} = {n_found_orig/n:.3f}")
    print(f"Found ANY chunk with correct answer in top-5 (upper bound, incl. paraphrase's own chunk): {n_found_any}/{n} = {n_found_any/n:.3f}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
