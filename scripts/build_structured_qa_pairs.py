"""
Synthetic (query, chunk_text) pairs for the 5 structured tables (CourseDetails,
FacultyList, Coordinator, Prerequisites, FacultyAvailability), which together
are ~2,140/5,059 = ~42% of the corpus but are NOT represented at all in the
(question, answer) pairs scripts/finetune_embeddings*.py trains and validates
on -- those come exclusively from EnglishQA/BanglishQA. Found as a likely
explanation for a real, measured gap (results/ir_metrics.csv, post-hard-
negative-deployment): the isolated held-out QA-pair embedding evaluation
showed a clean win for the mined-hard-negatives model, but full-corpus
vector-only retrieval showed a MIXED result (Recall@1 up, Recall@5/nDCG down
slightly) -- consistent with a model whose contrastive fine-tuning sharpened
discrimination specifically among QA-pair-style text, at some cost to how it
ranks the corpus's non-QA-pair chunks.

Each query is composed directly from the record's own structured fields
(same reference-construction convention as scripts/build_faculty_test_
queries.py and every other entity-heavy test set in this project -- not
scraped or invented text), with 1-3 natural phrasings per record where a
record supports more than one query shape. This is SYNTHETIC data,
disclosed as such -- these are not organically-collected questions, and this
script says so plainly rather than let them be mistaken for real user
queries if this file is read later.

TRAIN/VAL split: structured tables have no pre-existing Split column (unlike
EnglishQA/BanglishQA), so this script creates its own, at the RECORD level
(not the query level -- multiple queries about the same record all go to the
same split, so a val-split record's chunk is never seen in any form during
training), fixed seed=42, 85/15 train/val. Held out consistently with this
project's standing leakage rule: only Split='train' rows may be used for
fine-tuning; Split='val' rows are reserved for the held-out evaluation this
extension is meant to validate against (scripts/eval_structured_embeddings.py).

Usage: python scripts/build_structured_qa_pairs.py
"""

import json
import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "knowledge_base.db"
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
OUT_PATH = ROOT / "data" / "structured_qa_pairs.csv"
SEED = 42
VAL_FRACTION = 0.15


def _chunk_text_by_row(table_name: str):
    """doc_id -> chunk text, and row_id -> doc_id, both read directly from
    the corpus this project actually embeds (not reconstructed by hand), so
    the "positive" half of every pair is guaranteed to be the exact text
    ChromaDB indexes for that record."""
    row_to_doc = {}
    text_by_doc = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["metadata"].get("table") == table_name:
                row_id = rec["metadata"].get("row_id")
                row_to_doc[row_id] = rec["doc_id"]
                text_by_doc[rec["doc_id"]] = rec["text"]
    return row_to_doc, text_by_doc


def build_faculty_list_pairs(conn, rng):
    cur = conn.cursor()
    cur.execute("SELECT id, Name, Designation, Room, Email FROM FacultyList WHERE Name IS NOT NULL")
    rows = cur.fetchall()
    row_to_doc, text_by_doc = _chunk_text_by_row("FacultyList")
    pairs = []
    for row_id, name, designation, room, email in rows:
        doc_id = row_to_doc.get(str(row_id))
        if not doc_id:
            continue
        queries = []
        if room:
            queries.append(f"What is {name}'s office room?")
        if email:
            queries.append(f"What is {name}'s email address?")
        if designation:
            queries.append(f"What is {name}'s designation?")
        for q in queries:
            pairs.append({"table": "FacultyList", "row_id": row_id, "doc_id": doc_id,
                          "query": q, "chunk_text": text_by_doc[doc_id]})
    return pairs


def build_coordinator_pairs(conn, rng):
    cur = conn.cursor()
    cur.execute("SELECT id, Course FROM Coordinator WHERE Course IS NOT NULL")
    rows = cur.fetchall()
    row_to_doc, text_by_doc = _chunk_text_by_row("Coordinator")
    pairs = []
    for row_id, course in rows:
        doc_id = row_to_doc.get(str(row_id))
        if not doc_id:
            continue
        q = f"Who is the coordinator for {course}?"
        pairs.append({"table": "Coordinator", "row_id": row_id, "doc_id": doc_id,
                      "query": q, "chunk_text": text_by_doc[doc_id]})
    return pairs


def build_prerequisites_pairs(conn, rng):
    cur = conn.cursor()
    cur.execute("SELECT id, Course FROM Prerequisites WHERE Course IS NOT NULL")
    rows = cur.fetchall()
    row_to_doc, text_by_doc = _chunk_text_by_row("Prerequisites")
    pairs = []
    for row_id, course in rows:
        doc_id = row_to_doc.get(str(row_id))
        if not doc_id:
            continue
        q = f"What is the prerequisite for {course}?"
        pairs.append({"table": "Prerequisites", "row_id": row_id, "doc_id": doc_id,
                      "query": q, "chunk_text": text_by_doc[doc_id]})
    return pairs


def build_course_details_pairs(conn, rng):
    cur = conn.cursor()
    cur.execute("SELECT id, Course FROM CourseDetails WHERE Course IS NOT NULL")
    rows = cur.fetchall()
    row_to_doc, text_by_doc = _chunk_text_by_row("CourseDetails")
    pairs = []
    for row_id, course in rows:
        doc_id = row_to_doc.get(str(row_id))
        if not doc_id:
            continue
        q = f"What is the class schedule for {course}?"
        pairs.append({"table": "CourseDetails", "row_id": row_id, "doc_id": doc_id,
                      "query": q, "chunk_text": text_by_doc[doc_id]})
    return pairs


def build_faculty_availability_pairs(conn, rng):
    cur = conn.cursor()
    cur.execute("SELECT id, Name, Day FROM FacultyAvailability WHERE Name IS NOT NULL AND Day IS NOT NULL")
    rows = cur.fetchall()
    row_to_doc, text_by_doc = _chunk_text_by_row("FacultyAvailability")
    pairs = []
    for row_id, name, day in rows:
        doc_id = row_to_doc.get(str(row_id))
        if not doc_id:
            continue
        q = f"What is {name}'s schedule on {day}?"
        pairs.append({"table": "FacultyAvailability", "row_id": row_id, "doc_id": doc_id,
                      "query": q, "chunk_text": text_by_doc[doc_id]})
    return pairs


def main():
    conn = sqlite3.connect(DB_PATH)
    rng = random.Random(SEED)

    all_pairs = []
    all_pairs += build_faculty_list_pairs(conn, rng)
    all_pairs += build_coordinator_pairs(conn, rng)
    all_pairs += build_prerequisites_pairs(conn, rng)
    all_pairs += build_course_details_pairs(conn, rng)
    all_pairs += build_faculty_availability_pairs(conn, rng)
    conn.close()

    df = pd.DataFrame(all_pairs)

    # Record-level split (table, row_id) so every query about the same
    # record lands in the same split -- prevents a val-split record's chunk
    # text from ever being seen (via a different phrasing) during training.
    record_keys = sorted(df[["table", "row_id"]].drop_duplicates().itertuples(index=False, name=None))
    rng.shuffle(record_keys)
    n_val = max(1, round(len(record_keys) * VAL_FRACTION))
    val_keys = set(record_keys[:n_val])
    df["split"] = df.apply(lambda r: "val" if (r["table"], r["row_id"]) in val_keys else "train", axis=1)

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} synthetic structured-table pairs to {OUT_PATH}")
    print(df.groupby(["table", "split"]).size().unstack(fill_value=0))
    print(f"\nTotal unique records: {len(record_keys)}, val records: {len(val_keys)}")


if __name__ == "__main__":
    main()
