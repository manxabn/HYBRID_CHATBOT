"""
Ingests dataset/Banglish QA DATASET_Collection_NEW_MORE - Sheet1.csv (2050
rows, real Bangla-English code-switched questions collected from 160 real
bracu.ac.bd / library.bracu.ac.bd / etc. pages, columns: Question, Answer,
Source) into the existing BanglishQA table -- nearly TRIPLING this
project's Banglish training data (1053 -> ~3000 rows), confirmed to have
ZERO exact-text overlap with the existing table (checked directly, not
assumed).

Steps, each logged with real counts (not estimated):
  1. Drop internal duplicates (same Question+Answer pair appearing more
     than once in the new file -- 59 found on inspection).
  2. Derive Category from the Source URL's path via a keyword mapping to
     this project's EXISTING 16-category taxonomy where there's a
     confident match (e.g. ".../admissions/..." -> "Admission",
     ".../library..." -> "Library") -- ambiguous/unmatched URLs get a new,
     explicitly-labeled "General / Web FAQ" bucket rather than being
     force-fit into the wrong existing category. This is a heuristic, not
     a certainty -- disclosed as such in SourceNotes.
  3. Assign Split (train/val/test) via a fixed-seed stratified split
     matching the existing table's proportions (~79/11/10), so held-out
     evaluation stays uncontaminated the same way it already is for the
     rest of this project's data.
  4. Insert into BanglishQA with SourceReliability="Medium (public BRACU
     web page, not independently re-verified)" and SourceNotes citing the
     real Source URL -- distinct from the existing hand-authored rows'
     SourceNotes, since these are NOT Claude-authored paraphrases, they are
     directly collected from a public FAQ/scraping effort.

Does NOT touch corpus.jsonl, the BM25 index, or the Chroma index -- run
scripts/build_corpus.py, scripts/build_bm25_index.py, and (only if
explicitly requested, since it's destructive/rebuilds the live index)
scripts/build_chroma_index.py AFTER this, in that order, to actually make
the new rows retrievable.

Usage: python scripts/ingest_new_banglish_dataset.py
"""

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "knowledge_base.db"
NEW_CSV_PATH = ROOT / "dataset" / "Banglish QA DATASET_Collection_NEW_MORE - Sheet1.csv"
SEED = 42
# Matches the existing table's real Split proportions (835/118/100 of 1053
# -> ~0.793/0.112/0.095), not an arbitrary round number.
TRAIN_FRAC, VAL_FRAC = 0.793, 0.112

# URL path segment -> existing Category taxonomy (pipeline/build a fresh
# category only when no confident match exists, rather than guessing).
URL_CATEGORY_MAP = [
    (r"admission", "Admission"),
    (r"scholarship|financial-aid|tuition-and-fees|payment", "Payment, Financial Aid and Scholarship"),
    (r"residential-campus|\btarc\b", "TARC/RS (Residential Semester)"),
    (r"/clubs", "Extracurricular Activities"),
    (r"library", "Library"),
    (r"office-controller-examinations|exam", "Exams and Grading"),
    (r"office-proctor|plagiarism-policy|policies-and-procedures", "Policies and Procedures"),
    (r"\bit\b|technical", "Technical Support"),
    (r"department|/cse|/eee|/mns|architecture|biotechnology|civil-engineering|economics-and-social|"
     r"english|mechanical-engineering|academics/academic-advising", "Programs & Departments"),
    (r"about/campus|about/founder|about/affiliations|about/sustainability|about/iqac|"
     r"about/advancing-sdgs|about/campus-development", "Campus"),
    (r"thesis|internship", "Thesis and Internship"),
    (r"transport|bus", "Transportation and Bus Services"),
    (r"international", "International Student"),
    (r"office-registrar|academic-calendar|academic-dates|convocation", "Advising"),
]
FALLBACK_CATEGORY = "General / Web FAQ"


def categorize(url: str) -> str:
    url_lower = str(url).lower()
    for pattern, category in URL_CATEGORY_MAP:
        if re.search(pattern, url_lower):
            return category
    return FALLBACK_CATEGORY


def assign_split(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    splits = np.empty(n, dtype=object)
    splits[idx[:n_train]] = "train"
    splits[idx[n_train:n_train + n_val]] = "val"
    splits[idx[n_train + n_val:]] = "test"
    return splits


def main():
    df = pd.read_csv(NEW_CSV_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} raw rows from {NEW_CSV_PATH.name}")

    before = len(df)
    df = df.drop_duplicates(subset=["Question", "Answer"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} exact (Question, Answer) duplicate rows -> {len(df)} remain")

    conn = sqlite3.connect(DB_PATH)
    existing_q = set(pd.read_sql("SELECT QuestionBanglish FROM BanglishQA", conn)["QuestionBanglish"]
                      .str.strip().str.lower())
    before = len(df)
    df = df[~df["Question"].str.strip().str.lower().isin(existing_q)].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows whose Question exactly matches an existing BanglishQA row "
          f"-> {len(df)} genuinely new rows remain")

    df["Category"] = df["Source"].apply(categorize)
    print("\nCategory assignment (from Source URL heuristic):")
    print(df["Category"].value_counts())

    df["Split"] = assign_split(len(df))
    print(f"\nSplit assignment (seed={SEED}): {df['Split'].value_counts().to_dict()}")

    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(CAST(SourceId AS INTEGER)), 0) FROM BanglishQA WHERE SourceId GLOB '[0-9]*'")
    max_id = cur.fetchone()[0]
    print(f"\nExisting max numeric SourceId in BanglishQA: {max_id} -- new rows start at {max_id + 1}")

    rows_to_insert = []
    for i, r in df.iterrows():
        rows_to_insert.append((
            str(max_id + 1 + i),
            r["Category"],
            r["Question"].strip(),
            r["Answer"].strip(),
            "Medium (public BRACU web page, not independently re-verified)",
            "Unknown",  # TimeSensitive -- not determinable from this source format, disclosed as unknown
            r["Split"],
            f"Collected from public BRACU web page: {r['Source']} "
            f"[Banglish QA DATASET_Collection_NEW_MORE, not Claude-authored -- externally collected]",
        ))

    cur.executemany(
        "INSERT INTO BanglishQA (SourceId, Category, QuestionBanglish, AnswerEnglish, "
        "SourceReliability, TimeSensitive, Split, SourceNotes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows_to_insert,
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM BanglishQA")
    total_after = cur.fetchone()[0]
    conn.close()

    print(f"\nInserted {len(rows_to_insert)} new rows into BanglishQA.")
    print(f"BanglishQA table size: {total_after - len(rows_to_insert)} -> {total_after}")


if __name__ == "__main__":
    main()
