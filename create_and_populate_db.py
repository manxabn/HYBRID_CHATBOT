# create_and_populate_db.py
#
# Builds knowledge_base.db from the dataset/ CSVs. Sources, as of 2026-07-26:
#   EnglishQA          <- dataset/BRACU_QA_Dataset_FINAL.csv       (replaces the old "QUES - Sheet1.csv")
#   BanglishQA         <- dataset/BRACU_Banglish_QA_Only.csv       (new table)
#   CourseDetails      <- dataset/gsheet_course_routine.csv        (replaces "Tabular - Sheet1.csv")
#   FacultyList        <- dataset/gsheet_faculty_roster.csv        (replaces the old FacultyList CSV)
#   Coordinator        <- dataset/gsheet_coordinators_v2.csv       (replaces "Coordinator - Sheet1.csv")
#   FacultyAvailability<- dataset/gsheet_faculty_availability_raw/*.csv, via scripts/faculty_availability.py (new table)
#   Prerequisites      <- dataset/prerequisite - Sheet1.csv        (unchanged -- no newer source given)
#
# gsheet_course_routine.csv / gsheet_faculty_roster.csv / gsheet_coordinators_v2.csv /
# gsheet_faculty_availability_raw/*.csv were all pulled from
# https://docs.google.com/spreadsheets/d/1uCpxARIPFmkhL1BdzCL5dXmxO5CbNbFkKrgOCmUM6cA
# ("[forStudent] 2026 Summer CSE Routine / Consultation v1.0", confirmed via
# the sheet's own <title>) via the gviz CSV export endpoint. That workbook
# has 213 tabs total, and -- important -- the sheet's own TAB NAMES do not
# reliably match their content:
#   - tab literally named "FacultyList" (gid=2069304119) -> course routine grid -> CourseDetails
#   - tab literally named "Course_Coordinators" (gid=273249035) -> faculty roster -> FacultyList
#   - tab literally named "AAR" (gid=1073781740) -> Theory/Lab Coordinators table -> Coordinator
#   - the other 210 tabs, named by faculty initial, are personal weekly
#     consultation-hour grids -- but at least one of THOSE (tab "ZBYR")
#     actually belongs to initial "WLV". scripts/faculty_availability.py
#     parses the real Initial/Name/Email out of each tab's own cell content
#     rather than trusting the tab name, for exactly this reason.
# Filenames in this repo reflect actual content, not the sheet's tab labels.

import sqlite3
from pathlib import Path

import pandas as pd

from scripts.faculty_availability import parse_all as parse_faculty_availability

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"


def clean(value):
    """NaN/empty -> None so sqlite stores NULL instead of the literal string 'nan'."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def create_and_populate_db():
    # Both BRACU_*.csv files are Windows-1252, not UTF-8 (confirmed by byte
    # inspection: e.g. smart-quote byte 0x92 in the English file, bullet/
    # en-dash bytes 0x95/0x96 in the Banglish file) -- despite being the
    # "cleaned" exports, plain utf-8 decoding fails partway through both.
    # The gsheet_*.csv files are genuinely UTF-8 (fetched fresh, written as
    # UTF-8) and don't need this.
    english_qa_data = pd.read_csv(DATASET_DIR / "BRACU_QA_Dataset_FINAL.csv", encoding="cp1252")
    banglish_qa_data = pd.read_csv(DATASET_DIR / "BRACU_Banglish_QA_Only.csv", encoding="cp1252")

    course_routine_data = pd.read_csv(DATASET_DIR / "gsheet_course_routine.csv")
    # The sheet's header row has a literal newline inside "Theory Time\n(1hr
    # 20min)" and two trailing unlabeled columns (one genuinely unused, one
    # holding a contact email) -- rename positionally rather than trust the
    # raw header text. Fail loudly if the sheet's shape changes upstream.
    expected_cols = 13
    if len(course_routine_data.columns) != expected_cols:
        raise ValueError(
            f"gsheet_course_routine.csv has {len(course_routine_data.columns)} columns, "
            f"expected {expected_cols} -- the source sheet's layout changed; "
            "update the column mapping below before re-running."
        )
    course_routine_data.columns = [
        "Course", "TheoryEquivalent", "LabEquivalent", "TheoryInitial", "TheoryDay",
        "TheoryTime", "TheoryRoom", "LabFaculty", "LabDay", "LabTime", "LabRoom",
        "_Unused", "ContactEmail",
    ]

    faculty_data = pd.read_csv(DATASET_DIR / "gsheet_faculty_roster.csv")
    # gsheet_coordinators_v2.csv is a verbatim copy of the raw fetched tab,
    # which has a 2-row header: a "Theory Coordinators"/"Lab Coordinators"
    # section-label row, THEN the real column names ("Course","Theory 1",...).
    # Without skiprows=1, pandas treats the label row as the header and every
    # real row silently lands under "Unnamed: N" columns with no "Course"
    # column at all -- caught by inspecting actual DB row counts after a
    # first run inserted 0 real Coordinator rows despite looking fine at a
    # glance (see the row-count fix below, which is what surfaced this).
    coordinator_data = pd.read_csv(DATASET_DIR / "gsheet_coordinators_v2.csv", skiprows=1)
    prerequisite_data = pd.read_csv(DATASET_DIR / "prerequisite - Sheet1.csv")

    print("Parsing faculty availability grids (this may take a moment)...")
    availability_records, availability_skipped = parse_faculty_availability()
    print(f"  Parsed {len(availability_records)} (faculty, day) rows; "
          f"skipped {len(availability_skipped)} unparseable tabs")
    if availability_skipped:
        for gid, reason in availability_skipped:
            print(f"    skipped {gid}: {reason}")

    db_path = ROOT / "knowledge_base.db"
    if db_path.exists():
        # This function is a full rebuild from the CSVs (which are the
        # source of truth), and CREATE TABLE IF NOT EXISTS below would
        # otherwise silently keep any stale pre-existing schema (e.g. an
        # EnglishQA table from before BanglishQA/SourceId existed) and fail
        # confusingly on the INSERTs. Safe to remove: everything here is
        # reproducible from dataset/*.
        print(f"Removing existing {db_path} to rebuild from current schema")
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1) English QA (now BRACU_QA_Dataset_FINAL.csv, with real provenance columns)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS EnglishQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SourceId TEXT,
            Category TEXT,
            Question TEXT,
            Answer TEXT,
            Type TEXT,
            Register TEXT,
            SourceReliability TEXT,
            TimeSensitive TEXT,
            Split TEXT,
            SourceNotes TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2) Banglish QA (new table -- code-mixed Bangla/English questions, English answers)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BanglishQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SourceId TEXT,
            Category TEXT,
            QuestionBanglish TEXT,
            AnswerEnglish TEXT,
            SourceReliability TEXT,
            TimeSensitive TEXT,
            Split TEXT,
            SourceNotes TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3) Course Details (now the routine-grid tab; LabInitial -> LabFaculty
    #    since the new source gives possibly-multiple comma-separated
    #    initials, e.g. "SBHN,ESF", not a single one; added ContactEmail)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CourseDetails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Course TEXT,
            TheoryEquivalent TEXT,
            LabEquivalent TEXT,
            TheoryInitial TEXT,
            TheoryDay TEXT,
            TheoryTime TEXT,
            TheoryRoom TEXT,
            LabFaculty TEXT,
            LabDay TEXT,
            LabTime TEXT,
            LabRoom TEXT,
            ContactEmail TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 4) Faculty List (now the roster tab pulled from the live sheet)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FacultyList (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Initial TEXT,
            Name TEXT,
            Designation TEXT,
            Status TEXT,
            Room TEXT,
            Email TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 5) Coordinator Details (now the "Theory/Lab Coordinators" tab -- adds a
    #    3rd theory/lab coordinator slot and, unlike the old source, genuinely
    #    separate TheoryEmail/LabEmail instead of reusing one Email column
    #    for both. Covers 53 courses vs. the old source's 68 -- narrower,
    #    presumably current-term-active courses only; not a strict superset.)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Coordinator (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Course TEXT,
            FirstTheoryCoordinator TEXT,
            SecondTheoryCoordinator TEXT,
            ThirdTheoryCoordinator TEXT,
            TheoryEmail TEXT,
            FirstLabCoordinator TEXT,
            SecondLabCoordinator TEXT,
            ThirdLabCoordinator TEXT,
            LabEmail TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 6) Prerequisites (unchanged source)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Prerequisites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Course TEXT,
            PreRequisite TEXT,
            FullChainPreRequisite TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 7) Faculty Availability (new -- parsed from the 210 per-faculty
    #    consultation-hour grid tabs; one row per (faculty, day) that has at
    #    least one non-blank slot, see scripts/faculty_availability.py)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FacultyAvailability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Initial TEXT,
            Name TEXT,
            Programme TEXT,
            Email TEXT,
            Semester TEXT,
            Room TEXT,
            Day TEXT,
            ScheduleText TEXT,
            SourceGid TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Populate

    for _, row in english_qa_data.iterrows():
        cursor.execute('''
            INSERT INTO EnglishQA (
                SourceId, Category, Question, Answer, Type, Register,
                SourceReliability, TimeSensitive, Split, SourceNotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            clean(row.get("ID")), clean(row.get("Category")),
            clean(row.get("Question")), clean(row.get("Answer")),
            clean(row.get("Type")), clean(row.get("Register")),
            clean(row.get("Source Reliability")), clean(row.get("Time-Sensitive")),
            clean(row.get("Split")), clean(row.get("Source / Notes")),
        ))

    for _, row in banglish_qa_data.iterrows():
        cursor.execute('''
            INSERT INTO BanglishQA (
                SourceId, Category, QuestionBanglish, AnswerEnglish,
                SourceReliability, TimeSensitive, Split, SourceNotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            clean(row.get("ID")), clean(row.get("Category")),
            clean(row.get("Question (Banglish)")), clean(row.get("Answer (English)")),
            clean(row.get("Source Reliability")), clean(row.get("Time-Sensitive")),
            clean(row.get("Split")), clean(row.get("Source / Notes")),
        ))

    for _, row in course_routine_data.iterrows():
        course = clean(row.get("Course"))
        if not course:
            continue  # trailing blank rows some sheet exports include
        cursor.execute('''
            INSERT INTO CourseDetails (
                Course, TheoryEquivalent, LabEquivalent,
                TheoryInitial, TheoryDay, TheoryTime, TheoryRoom,
                LabFaculty, LabDay, LabTime, LabRoom, ContactEmail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            course, clean(row.get("TheoryEquivalent")), clean(row.get("LabEquivalent")),
            clean(row.get("TheoryInitial")), clean(row.get("TheoryDay")),
            clean(row.get("TheoryTime")), clean(row.get("TheoryRoom")),
            clean(row.get("LabFaculty")), clean(row.get("LabDay")),
            clean(row.get("LabTime")), clean(row.get("LabRoom")),
            clean(row.get("ContactEmail")),
        ))

    for _, row in faculty_data.iterrows():
        initial = clean(row.get("Initial"))
        if not initial:
            continue
        cursor.execute('''
            INSERT INTO FacultyList (Initial, Name, Designation, Status, Room, Email)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            initial, clean(row.get("Name")), clean(row.get("Designation")),
            clean(row.get("Status")), clean(row.get("Room")), clean(row.get("Email")),
        ))

    for _, row in coordinator_data.iterrows():
        course = clean(row.get("Course"))
        if not course:
            continue
        cursor.execute('''
            INSERT INTO Coordinator (
                Course, FirstTheoryCoordinator, SecondTheoryCoordinator, ThirdTheoryCoordinator,
                TheoryEmail, FirstLabCoordinator, SecondLabCoordinator, ThirdLabCoordinator, LabEmail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            course, clean(row.get("Theory 1")), clean(row.get("Theory 2")), clean(row.get("Theory 3")),
            clean(row.get("Theory Email")),
            clean(row.get("Lab 1")), clean(row.get("Lab 2")), clean(row.get("Lab 3")),
            clean(row.get("Lab Email")),
        ))

    for _, row in prerequisite_data.iterrows():
        cursor.execute('''
            INSERT INTO Prerequisites (
                Course, PreRequisite, FullChainPreRequisite
            ) VALUES (?, ?, ?)
        ''', (
            clean(row.get("Course")), clean(row.get("Pre-Requisite")),
            clean(row.get("Full Chain Pre-Requisite")),
        ))

    for rec in availability_records:
        cursor.execute('''
            INSERT INTO FacultyAvailability (
                Initial, Name, Programme, Email, Semester, Room, Day, ScheduleText, SourceGid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rec["Initial"], rec["Name"], rec["Programme"], rec["Email"],
            rec["Semester"], rec["Room"], rec["Day"], rec["ScheduleText"], rec["SourceGid"],
        ))

    conn.commit()
    print("Database has been created and populated successfully.")
    # Report ACTUAL inserted rows via COUNT(*), not source-dataframe length --
    # a row that fails its "if not X: continue" guard (or, as happened with
    # the Coordinator source once, an entire misread file) silently inserts
    # nothing while len(dataframe) still looks fine. This is what would have
    # caught the gsheet_coordinators_v2.csv header bug immediately instead of
    # requiring a manual SELECT COUNT(*) to notice.
    for table, source_len in [
        ("EnglishQA", len(english_qa_data)), ("BanglishQA", len(banglish_qa_data)),
        ("CourseDetails", len(course_routine_data)), ("FacultyList", len(faculty_data)),
        ("Coordinator", len(coordinator_data)), ("Prerequisites", len(prerequisite_data)),
        ("FacultyAvailability", len(availability_records)),
    ]:
        actual = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        flag = "  <-- MISMATCH, check source columns/parsing" if actual != source_len else ""
        print(f"  {table:20s} {actual} rows in DB (source had {source_len} candidate rows){flag}")
    conn.close()


if __name__ == "__main__":
    create_and_populate_db()
