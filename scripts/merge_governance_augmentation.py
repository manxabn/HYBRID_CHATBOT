"""
Merges data/governance_augmentation_proposed.csv (88 new paraphrase rows,
22 already-verified "University Facts & Governance" facts x 4 diversity-
filtered paraphrases each) into knowledge_base.db's EnglishQA table --
user-approved 2026-07-31 after reviewing a sample of the proposed rows.

Verified before merge (see conversation, not re-derived here):
  - All 22 source facts exist verbatim in EnglishQA (100% matched by
    Question text).
  - Every proposed row's answer field is byte-identical to its matched
    source fact's Answer in the DB (0/22 mismatches) -- confirms this
    augmentation only adds new PHRASINGS of already-verified facts, never
    new factual content.

Each inserted row inherits Type/Register/SourceReliability/TimeSensitive
from its matched source fact (the underlying fact's provenance is
unchanged by paraphrasing it), except:
  - SourceId: the proposed row's own query_id (e.g. GOV-AUG-000-0), for
    traceability back to data/governance_augmentation_proposed.csv.
  - Register: prefixed to disclose LLM-paraphrase origin rather than
    silently inheriting a register that would misrepresent these as
    directly student-submitted or independently-sourced text.
  - Split: the proposed row's own proposed_split (target 49/20/19 train/
    val/test, addressing the original 2/22-in-val, 3/22-in-test coverage
    gap this augmentation was built to fix).
  - SourceNotes: original notes plus a back-reference to the source fact's
    own SourceId, so any future audit can trace a paraphrase to the exact
    verified fact it was generated from.

Usage: python scripts/merge_governance_augmentation.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "knowledge_base.db"
CSV_PATH = ROOT / "data" / "governance_augmentation_proposed.csv"


def main():
    prop = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM EnglishQA WHERE Category='University Facts & Governance'")
    before_count = cur.fetchone()[0]
    cur.execute("SELECT Split, COUNT(*) FROM EnglishQA WHERE Category='University Facts & Governance' GROUP BY Split")
    before_split = dict(cur.fetchall())
    print(f"Before merge: {before_count} governance rows, split={before_split}")

    source_meta = {}
    for s in prop["source_question"].unique():
        cur.execute(
            "SELECT SourceId, Answer, Type, Register, SourceReliability, TimeSensitive, SourceNotes "
            "FROM EnglishQA WHERE Question=?", (s,))
        row = cur.fetchone()
        if row is None:
            print(f"FATAL: source question not found, aborting without writing anything: {s!r}")
            conn.close()
            sys.exit(1)
        src_id, answer, type_, register, reliability, time_sensitive, notes = row
        source_meta[s] = {
            "src_id": src_id, "answer": answer, "type": type_, "register": register,
            "reliability": reliability, "time_sensitive": time_sensitive, "notes": notes,
        }

    inserted = 0
    for _, r in prop.iterrows():
        meta = source_meta[r["source_question"]]
        if str(meta["answer"]).strip() != str(r["answer"]).strip():
            print(f"FATAL: answer drift detected for {r['query_id']}, aborting without writing anything")
            conn.close()
            sys.exit(1)
        cur.execute(
            "INSERT INTO EnglishQA (SourceId, Category, Question, Answer, Type, Register, "
            "SourceReliability, TimeSensitive, Split, SourceNotes) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                r["query_id"],
                r["category"],
                r["question"],
                r["answer"],
                meta["type"],
                f"LLM-generated paraphrase (diversity-filtered) of a {meta['register']} fact",
                meta["reliability"],
                meta["time_sensitive"],
                r["proposed_split"],
                f"{meta['notes']} | Paraphrase-augmented from source fact SourceId={meta['src_id']} "
                f"via scripts/augment_governance_category.py, merged 2026-07-31",
            ),
        )
        inserted += 1

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM EnglishQA WHERE Category='University Facts & Governance'")
    after_count = cur.fetchone()[0]
    cur.execute("SELECT Split, COUNT(*) FROM EnglishQA WHERE Category='University Facts & Governance' GROUP BY Split")
    after_split = dict(cur.fetchall())
    print(f"Inserted {inserted} rows.")
    print(f"After merge: {after_count} governance rows, split={after_split}")
    conn.close()


if __name__ == "__main__":
    main()
