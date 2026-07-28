"""
Builds a test set for FacultyAvailability, the one corpus table paper.tex
explicitly disclosed as untested (814 records, ingested and retrievable but
never the source of a scored query). Two query shapes, both answerable
directly from the table's own fields (same reference-construction
convention as the Prerequisites/Coordinator/CourseDetails entity-heavy
queries elsewhere in this project -- composed directly from structured
fields, not copied verbatim from a corpus chunk):

  - "office room" queries: room is the same across every day-row for a
    given faculty member, so this exercises retrieval robustness to the
    table's one-row-per-day chunking (any of that faculty's ~5-6 chunks
    would answer it correctly).
  - "day schedule" queries: which day is also part of what identifies the
    correct chunk (this faculty's OTHER days' chunks are near-identical
    distractors, differing only in the Day/ScheduleText fields) -- a
    stricter test than the room query.

A fixed seed samples unique faculty (by Initial) so the test set doesn't
just repeat one person's five day-rows.

Usage: python scripts/build_faculty_test_queries.py
"""

import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "data" / "test_queries_faculty.csv"
SEED = 42
N_FACULTY_FOR_ROOM = 30
N_DAY_QUERIES = 20


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT Initial, Name, Room FROM FacultyAvailability WHERE Room IS NOT NULL AND Room != ''")
    faculty = cur.fetchall()
    cur.execute("SELECT Initial, Name, Day, ScheduleText FROM FacultyAvailability "
                "WHERE ScheduleText IS NOT NULL AND ScheduleText != ''")
    day_rows = cur.fetchall()
    conn.close()

    rng = random.Random(SEED)

    records = []
    room_sample = rng.sample(faculty, min(N_FACULTY_FOR_ROOM, len(faculty)))
    for i, (initial, name, room) in enumerate(room_sample):
        records.append({
            "query_id": f"FAC-R{i+1:03d}",
            "query": f"What room is {name}'s office in?",
            "reference_answer": f"{name}'s office is in room {room}.",
            "query_type": "room",
        })

    day_sample = rng.sample(day_rows, min(N_DAY_QUERIES, len(day_rows)))
    for i, (initial, name, day, schedule) in enumerate(day_sample):
        records.append({
            "query_id": f"FAC-D{i+1:03d}",
            "query": f"What is {name}'s schedule on {day}?",
            "reference_answer": f"On {day}, {name} is scheduled: {schedule}.",
            "query_type": "day_schedule",
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} FacultyAvailability test queries to {OUT_PATH} "
          f"({len(room_sample)} room, {len(day_sample)} day-schedule)")


if __name__ == "__main__":
    main()
