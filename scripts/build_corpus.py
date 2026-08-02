"""
Build a single unified corpus (data/corpus.jsonl) from knowledge_base.db.

Reads all 6 SQLite tables, chunks each record's combined text the same way
db_ingestion.py already does (500 char window / 50 char overlap), and writes
one JSONL file that both the Chroma index build and the BM25 index build
consume, so the two retrieval streams operate over identical (doc_id, text)
pairs.

Each table's INGESTION_PLAN entry splits fields into:
  - "text":       joined into the embedded/BM25-indexed chunk text
  - "labels":     human-readable label per "text" field, same order
  - "meta_extra": stored in metadata only, NOT embedded

This matters most for EnglishQA/BanglishQA: their source CSVs carry real
provenance columns (Source Reliability, Time-Sensitive, Split, ...) that are
useful to keep queryable but would otherwise pollute retrieval with strings
like "train" / "Medium (internal dataset...)" if joined into the chunk text
alongside the actual question/answer content.

Chunk text is written as "Label: value" per field, not bare values. Confirmed
live (2026-07-26 demo run) that bare-value text breaks multi-value structured
rows: Coordinator's chunk for CSE330 read "CSE330 AMK AQU ... NNTN ALB ...",
and the LLM -- correctly, given that input -- said it couldn't tell which
name was the theory vs. lab coordinator. Labeling fixes this at the source
rather than asking the LLM to guess field identity from position.
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "data" / "corpus.jsonl"

from pipeline.prerequisite_graph import PrerequisiteGraph

INGESTION_PLAN = {
    "EnglishQA": {
        "text": ["Question", "Answer"],
        "labels": ["Question", "Answer"],
        "meta_extra": ["SourceId", "Category", "Type", "Register",
                        "SourceReliability", "TimeSensitive", "Split", "SourceNotes"],
    },
    "BanglishQA": {
        "text": ["QuestionBanglish", "AnswerEnglish"],
        "labels": ["Question (Banglish)", "Answer (English)"],
        "meta_extra": ["SourceId", "Category", "SourceReliability",
                        "TimeSensitive", "Split", "SourceNotes"],
    },
    "CourseDetails": {
        "text": [
            "Course", "TheoryEquivalent", "LabEquivalent",
            "TheoryInitial", "TheoryDay", "TheoryTime", "TheoryRoom",
            "LabFaculty", "LabDay", "LabTime", "LabRoom",
        ],
        "labels": [
            "Course", "Theory Equivalent", "Lab Equivalent",
            "Theory Initial", "Theory Day", "Theory Time", "Theory Room",
            "Lab Faculty", "Lab Day", "Lab Time", "Lab Room",
        ],
        "meta_extra": ["ContactEmail"],
    },
    "FacultyList": {
        "text": ["Initial", "Name", "Designation", "Status", "Room", "Email"],
        "labels": ["Initial", "Name", "Designation", "Status", "Room", "Email"],
        "meta_extra": [],
    },
    "Coordinator": {
        "text": [
            "Course", "FirstTheoryCoordinator", "SecondTheoryCoordinator", "ThirdTheoryCoordinator",
            "TheoryEmail", "FirstLabCoordinator", "SecondLabCoordinator", "ThirdLabCoordinator", "LabEmail",
        ],
        "labels": [
            "Course", "First Theory Coordinator", "Second Theory Coordinator", "Third Theory Coordinator",
            "Theory Coordinator Email", "First Lab Coordinator", "Second Lab Coordinator",
            "Third Lab Coordinator", "Lab Coordinator Email",
        ],
        "meta_extra": [],
    },
    "Prerequisites": {
        "text": ["Course", "PreRequisite", "FullChainPreRequisite"],
        "labels": ["Course", "Prerequisite", "Full Prerequisite Chain"],
        "meta_extra": [],
    },
    "FacultyAvailability": {
        "text": ["Initial", "Name", "Day", "ScheduleText"],
        "labels": ["Initial", "Name", "Day", "Scheduled commitments (classes/labs/breaks, not availability)"],
        "meta_extra": ["Programme", "Email", "Semester", "Room", "SourceGid"],
    },
}


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 50):
    # chunk_size was 500 chars, uniformly for every table. Measured directly
    # against this corpus (2026-07-27): 11.4% of EnglishQA rows and 9.0% of
    # BanglishQA rows exceed 500 chars once formatted as "Question: ...\n
    # Answer: ...", so those rows were being fragmented into multiple
    # separately-indexed chunks -- with no guarantee retrieval brings back
    # every fragment of one row's answer together (confirmed concretely: a
    # multi-bullet-point answer was retrieved and generated truncated
    # mid-list). Chunking is per-row (each record's own combined_text is
    # chunked independently, see main() below), so a larger chunk_size can
    # only reduce/eliminate this fragmentation -- it never mixes unrelated
    # rows together. 2000 chars covers this corpus's measured p99 (~1350
    # chars) and max (~1900 chars) with margin, so effectively every row
    # becomes one intact chunk. Matches 2026 RAG chunking guidance for
    # QA-with-reranker setups (800-2048 tokens), which this pipeline is.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2026-08-01: the Prerequisites table's own stored FullChainPreRequisite
    # field is a data-quality bug, not a display-formatting one -- verified
    # directly against the DB: CSE220's own row lists CSE230 as a direct
    # prerequisite (PreRequisite='CSE111 (HP),CSE230 (HP)'), but CSE221's
    # stored "full chain" (which starts at CSE220) is 'CSE220-CSE111-CSE110'
    # -- CSE230 silently dropped one hop in, even though it's right there in
    # CSE220's own row. This propagates to everything downstream of CSE221
    # (CSE310, CSE471) and similarly drops PHY111/PHY112 from CSE260's chain
    # (and everything downstream: CSE340, CSE460); CSE341's stored field is
    # simply NULL. Confirmed via a full audit (2026-08-01): 7 of the 12
    # "full prerequisite chain" test queries have a reference_answer (built
    # from PrerequisiteGraph.full_chain()'s real BFS, in scripts/build_test_
    # queries.py) that requires a course code absent from this stale stored
    # field -- meaning the retrievable corpus chunk could NEVER satisfy
    # those queries' own ground truth, deflating every IR metric (recall/
    # MRR/nDCG) uniformly across every retrieval config by the same fixed,
    # avoidable amount. Fixed by overriding FullChainPreRequisite with the
    # SAME graph BFS already used to generate those reference answers, so
    # corpus content and ground truth are computed from the same source of
    # truth (the Prerequisites table's PreRequisite field via PrerequisiteGraph),
    # not two independently-drifted copies of the same fact. Deliberately
    # fixed only in the generated corpus, not by UPDATE-ing knowledge_base.db's
    # own stored field -- lower blast radius, and this is the only place
    # that field's value is actually read (grep-confirmed).
    prereq_graph = PrerequisiteGraph(db_path=DB_PATH)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_chunks = 0
    n_prereq_chains_fixed = 0
    per_table_chunks = {}

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for table_name, plan in INGESTION_PLAN.items():
            text_fields = plan["text"]
            labels = plan["labels"]
            meta_extra_fields = plan["meta_extra"]
            all_fields = text_fields + meta_extra_fields

            query = f"SELECT id, {', '.join(all_fields)} FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()

            n_text = len(text_fields)
            for row in rows:
                row_id = row[0]
                text_values = list(row[1:1 + n_text])
                meta_extra_values = row[1 + n_text:]

                if table_name == "Prerequisites":
                    chain_field_idx = text_fields.index("FullChainPreRequisite")
                    course_code = text_values[text_fields.index("Course")]
                    computed_chain = prereq_graph.full_chain(course_code)
                    computed_chain_str = "-".join(computed_chain) if computed_chain else None
                    if computed_chain_str != text_values[chain_field_idx]:
                        n_prereq_chains_fixed += 1
                    text_values[chain_field_idx] = computed_chain_str

                combined_text = "\n".join(
                    f"{label}: {value}" for label, value in zip(labels, text_values) if value
                )
                if not combined_text.strip():
                    continue

                doc_chunks = chunk_text(combined_text, chunk_size=2000, overlap=50)
                n_records += 1

                for chunk_i, chunk_content in enumerate(doc_chunks):
                    doc_id = f"{table_name}-{row_id}-chunk{chunk_i}"
                    meta = {"table": table_name, "row_id": str(row_id), "chunk_index": chunk_i}
                    for col_name, value in zip(text_fields, text_values):
                        meta[col_name] = str(value) if value is not None else ""
                    for col_name, value in zip(meta_extra_fields, meta_extra_values):
                        meta[col_name] = str(value) if value is not None else ""

                    out.write(json.dumps({
                        "doc_id": doc_id,
                        "table": table_name,
                        "text": chunk_content,
                        "metadata": meta,
                    }) + "\n")
                    n_chunks += 1
                    per_table_chunks[table_name] = per_table_chunks.get(table_name, 0) + 1

    conn.close()
    print(f"Read {n_records} non-empty records from {DB_PATH}")
    print(f"Wrote {n_chunks} chunks to {OUT_PATH}")
    for table_name, count in per_table_chunks.items():
        print(f"  {table_name}: {count} chunks")
    print(f"Prerequisites: {n_prereq_chains_fixed} FullChainPreRequisite values differed from the "
          f"DB's stale stored field and were overridden with the graph-computed BFS chain")


if __name__ == "__main__":
    main()
