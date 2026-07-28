"""
Parser for the 210 per-faculty "Consultation Hour Planner" tabs fetched from
https://docs.google.com/spreadsheets/d/1uCpxARIPFmkhL1BdzCL5dXmxO5CbNbFkKrgOCmUM6cA
into dataset/gsheet_faculty_availability_raw/{gid}.csv (raw CSV per tab,
fetched via the gviz export endpoint -- see create_and_populate_db.py).

IMPORTANT: the sheet's own tab names (e.g. "ZBYR") do NOT reliably match the
faculty inside the tab -- one sampled tab named "ZBYR" actually belongs to
initial "WLV" (Md. Waseq Alauddin Alvi). Every field here is parsed out of
each tab's own cell content, never out of the tab name/filename.

Two layout variants were found by inspecting all 211 fetched files:
  - Variant A (209 of 211): a single squashed header row where Google's CSV
    export flattened several merged cells together, e.g. column 3 literally
    reads "Mr. Abid Jahan Apon CSE abid.jahan@bracu.ac.bd 11:00 AM" -- name,
    programme, email, and the next column's time label all concatenated.
    9 time slots (8:00 AM through 7:30 PM).
  - Variant B (1 of 211, gid=1081681722 / faculty initial ACH): a cleaner
    multi-row header block (separate "Instructor:"/"Programme:"/"Email:"
    labeled rows) and 10 time slots (8:30 AM through 9:00 PM, an extra slot
    vs. Variant A).

One additional fetched tab (gid=1073781740, labeled "AAR") is NOT a
consultation grid at all -- it's an updated Theory/Lab Coordinators table.
It is excluded from FACULTY_TAB_GIDS below and handled separately as the new
Coordinator source (dataset/gsheet_coordinators_v2.csv); see
create_and_populate_db.py.

Parsing failures are reported, never guessed: if a tab's metadata (Initial/
Name/Email) can't be confidently extracted, that tab is skipped and logged,
not inserted with placeholder values.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "gsheet_faculty_availability_raw"
NON_FACULTY_GIDS = {"1073781740"}  # the Coordinators-v2 table, see module docstring

DAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME_SLOTS_A = ["8:00 AM", "9:30 AM", "11:00 AM", "12:30 PM", "2:00 PM",
                "3:30 PM", "5:00 PM", "6:00 PM", "7:30 PM"]
TIME_SLOTS_B = ["8:30 AM", "9:30 AM", "11:00 AM", "12:30 PM", "2:00 PM",
                "3:30 PM", "5:00 PM", "6:00 PM", "7:30 PM", "9:00 PM"]

EMAIL_RE = re.compile(r"[\w.\-]+@[\w\-]+\.[\w.\-]+(?:,\s*[\w.\-]+@[\w\-]+\.[\w.\-]+)*")
# The trailing optional (?:\s+\S+)? groups below tolerate a handful of tabs
# where the source sheet's own merged-cell formula duplicated the last field
# (e.g. "...mohammad.naveed@bracu.ac.bd mohammad.naveed@bracu.ac.bd 11:00 AM",
# "...4M113 4M113 3:30 PM") -- confirmed against gid=1727828495. Harmless for
# the normal (non-duplicated) case: greedy matching backtracks to zero-width
# once the real time-label suffix is found, so `email`/`room` still capture
# the correct single value either way.
NAME_PROGRAMME_EMAIL_RE = re.compile(
    r"^(?P<name>.+?)\s+(?P<programme>[A-Z]{2,6}(?:,\s*[A-Z]{2,6})*)\s+"
    r"(?P<email>[\w.\-]+@[\w\-]+\.[\w.\-]+)(?:\s+\S+)?\s+\d{1,2}:\d{2}\s*(AM|PM)$"
)
INITIAL_SEMESTER_ROOM_RE = re.compile(
    r"^(?P<initial>\S+)\s+(?P<semester>(?:Summer|Fall|Spring)\s+\d{4})\s+"
    r"(?P<room>\S+)(?:\s+\S+)?\s+\d{1,2}:\d{2}\s*(AM|PM)$"
)


def _read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def _clean_cell(text: str) -> str:
    return " ".join(text.split()).strip()


def _parse_variant_a(rows, gid):
    if len(rows) < 2:
        return None
    header = rows[0]
    if len(header) < 7:
        return None

    m_name = NAME_PROGRAMME_EMAIL_RE.match(header[3].strip())
    m_init = INITIAL_SEMESTER_ROOM_RE.match(header[6].strip())
    if not m_name or not m_init:
        return None

    meta = {
        "Initial": m_init.group("initial"),
        "Name": m_name.group("name").strip(),
        "Programme": m_name.group("programme"),
        "Email": m_name.group("email"),
        "Semester": m_init.group("semester"),
        "Room": m_init.group("room"),
        "SourceGid": gid,
    }

    days_out = []
    for row in rows[1:]:
        if not row or row[0].strip() not in DAYS:
            continue
        day = row[0].strip()
        slots = row[1:10]
        if len(slots) < len(TIME_SLOTS_A):
            slots = slots + [""] * (len(TIME_SLOTS_A) - len(slots))
        entries = [
            (t, _clean_cell(c)) for t, c in zip(TIME_SLOTS_A, slots) if _clean_cell(c)
        ]
        if entries:
            days_out.append((day, entries))

    return meta, days_out


def _parse_variant_b(rows, gid):
    # Rows are fixed-position for this layout (see module docstring).
    if len(rows) < 14:
        return None
    try:
        name = rows[2][3].strip()
        programme = rows[3][3].strip()
        email_cell = rows[4][2].strip()
        email_match = EMAIL_RE.search(email_cell)
        initial = rows[2][6].strip()
        semester = rows[3][6].strip()
        room = rows[4][6].strip()
    except IndexError:
        return None

    if not (name and initial and email_match):
        return None

    meta = {
        "Initial": initial,
        "Name": name,
        "Programme": programme,
        "Email": email_match.group(0),
        "Semester": semester,
        "Room": room,
        "SourceGid": gid,
    }

    days_out = []
    for row in rows[7:14]:
        if not row or row[0].strip() not in DAYS:
            continue
        day = row[0].strip()
        slots = row[1:11]
        if len(slots) < len(TIME_SLOTS_B):
            slots = slots + [""] * (len(TIME_SLOTS_B) - len(slots))
        entries = [
            (t, _clean_cell(c)) for t, c in zip(TIME_SLOTS_B, slots) if _clean_cell(c)
        ]
        if entries:
            days_out.append((day, entries))

    return meta, days_out


def parse_all():
    """Returns (records, skipped) where records is a list of dicts:
    {Initial, Name, Programme, Email, Semester, Room, Day, ScheduleText, SourceGid}
    -- one row per (faculty, day) with at least one non-blank slot.
    skipped is a list of (gid, reason) for tabs that couldn't be parsed."""
    records = []
    skipped = []

    files = sorted(RAW_DIR.glob("*.csv"))
    for path in files:
        gid = path.stem
        if gid in NON_FACULTY_GIDS:
            continue

        rows = _read_rows(path)
        if not rows:
            skipped.append((gid, "empty file"))
            continue

        is_variant_b = (
            len(rows[0]) > 1 and rows[0][1].strip() == "BRAC University"
        )
        parsed = _parse_variant_b(rows, gid) if is_variant_b else _parse_variant_a(rows, gid)

        if parsed is None:
            skipped.append((gid, "variant_b metadata regex mismatch" if is_variant_b
                             else "variant_a metadata regex mismatch"))
            continue

        meta, days_out = parsed
        if not days_out:
            skipped.append((gid, "parsed OK but zero non-blank slots"))
            continue

        for day, entries in days_out:
            schedule_text = "; ".join(f"{t} - {c}" for t, c in entries)
            records.append({
                **meta,
                "Day": day,
                "ScheduleText": schedule_text,
            })

    return records, skipped


if __name__ == "__main__":
    recs, skipped = parse_all()
    print(f"Parsed {len(recs)} (faculty, day) rows from "
          f"{len(list(RAW_DIR.glob('*.csv'))) - len(NON_FACULTY_GIDS)} candidate tabs")
    print(f"Skipped {len(skipped)} tabs:")
    for gid, reason in skipped:
        print(f"  {gid}: {reason}")
