# db_ingestion.py

import sqlite3
from knowledgebase import KnowledgeBase
from datetime import datetime

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split 'text' into chunks of length 'chunk_size' with 'overlap' between them.
    This helps embedding larger texts in smaller pieces.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)  # move start by chunk_size - overlap
    return chunks

class DBIngestion:
    def __init__(self, db_path="knowledge_base.db"):
        self.db_path = db_path
        self.kb = KnowledgeBase()
        # Store the last ingested timestamps in memory (key = table_name)
        self.last_ingested_timestamps = {
            "EnglishQA": None,
            "BanglishQA": None,
            "CourseDetails": None,
            "FacultyList": None,
            "Coordinator": None,
            "Prerequisites": None,
            "FacultyAvailability": None,
        }

    def ingest_new_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Define which fields to pull from each table for embedding
        # We'll also store them in metadata so the agent can see them if needed.
        # NOTE: unlike scripts/build_corpus.py (the pipeline actually used
        # for the ablation study), this simple ingestion_plan has no
        # text-vs-metadata split -- every listed field goes into both the
        # embedded text and the metadata. Keep EnglishQA/BanglishQA limited
        # to the actual question/answer fields so provenance strings (e.g.
        # "Medium (internal dataset...)") don't get embedded as if they were
        # content.
        ingestion_plan = {
            "EnglishQA": ["Question", "Answer"],
            "BanglishQA": ["QuestionBanglish", "AnswerEnglish"],
            "CourseDetails": [
                "Course", "TheoryEquivalent", "LabEquivalent",
                "TheoryInitial", "TheoryDay", "TheoryTime", "TheoryRoom",
                "LabFaculty", "LabDay", "LabTime", "LabRoom", "ContactEmail"
            ],
            "FacultyList": ["Initial", "Name", "Designation", "Status", "Room", "Email"],
            "Coordinator": [
                "Course", "FirstTheoryCoordinator", "SecondTheoryCoordinator", "ThirdTheoryCoordinator",
                "TheoryEmail", "FirstLabCoordinator", "SecondLabCoordinator", "ThirdLabCoordinator", "LabEmail"
            ],
            "Prerequisites": ["Course", "PreRequisite", "FullChainPreRequisite"],
            "FacultyAvailability": ["Initial", "Name", "Day", "ScheduleText"],
        }

        for table_name, fields in ingestion_plan.items():
            last_ts = self.last_ingested_timestamps.get(table_name)

            # If we have a last timestamp, only get rows newer than that
            if last_ts:
                query = f"""
                    SELECT id, {', '.join(fields)}, Timestamp
                    FROM {table_name}
                    WHERE Timestamp > ?
                """
                cursor.execute(query, (last_ts,))
            else:
                # Otherwise ingest all rows
                query = f"""
                    SELECT id, {', '.join(fields)}, Timestamp
                    FROM {table_name}
                """
                cursor.execute(query)

            rows = cursor.fetchall()
            if not rows:
                continue

            # For each row, create a doc for Chroma
            for row_data in rows:
                row_id = row_data[0]
                record_fields = row_data[1:-1]  # everything except ID and Timestamp
                timestamp_str = row_data[-1]

                # Combine the fields into text for embedding:
                # Combine fields
                combined_text = "\n".join(str(f) for f in record_fields if f)

                # Chunk the text
                doc_chunks = chunk_text(combined_text, chunk_size=500, overlap=50)

                # For each chunk, create a doc in the knowledge base
                for chunk_i, chunk_content in enumerate(doc_chunks):
                    doc_id = f"{table_name}-{row_id}-chunk{chunk_i}"

                    meta = {
                        "table": table_name,
                        "row_id": str(row_id),
                        "timestamp": timestamp_str,
                        "chunk_index": chunk_i
                    }
                    # store each field in metadata
                    for col_name, value in zip(fields, record_fields):
                        meta[col_name] = str(value) if value is not None else ""

                    self.kb.add_document(
                        doc_id=doc_id,
                        text=chunk_content,
                        meta=meta
                    )

            # Update the last_ingested_timestamps
            latest_ts = max(row[-1] for row in rows)
            self.last_ingested_timestamps[table_name] = latest_ts

        conn.close()
        # Persist changes
        self.kb.persist()
        print("Ingestion process complete. New data has been embedded and stored.")

    def run_ingestion(self):
        self.ingest_new_data()