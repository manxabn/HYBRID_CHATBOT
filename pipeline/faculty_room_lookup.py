"""
Course -> theory instructor -> office room cross-reference, the same
architectural pattern as pipeline/prerequisite_graph.py: a query shape
that needs TWO sequential structured-table lookups, which no single
retrieval pass can satisfy because the second fact (the instructor's own
FacultyList row: Name/Room/Designation/Email) shares almost no vocabulary
with the original query text or the CourseDetails chunk that names them
only by initial.

Motivation, found directly via scripts/test_compound_queries_expanded.py
(2026-07-31): for 540 real "who teaches {course} and what is their office
room?" queries, BOTH bm25_only and full_hybrid retrieval returned the
correct CourseDetails chunk (rank 1, reliably) but NEVER surfaced the
matching FacultyList chunk in the top 10 -- 0/540 for both configs. This
is not a fusion-method problem (hybrid and BM25 failed identically); it's
a structural retrieval-scope problem, the same class the prerequisite
graph was built to solve for multi-hop chains.

Deliberately conservative, matching this project's standing "verify, don't
guess" discipline: only fires when the query's course reference resolves
to EXACTLY ONE CourseDetails row. A bare base code (e.g. "CSE101") maps to
several sections, each potentially with a different instructor -- firing
on an ambiguous base code would require guessing which section the user
meant, which this project's ambiguous-entity handling elsewhere (Section
on unambiguous-match) explicitly avoids. If the query doesn't specify a
section AND more than one section exists, this returns None and the
caller falls back to plain retrieval, unchanged from today.

Usage (mirrors PrerequisiteGraph.context_block):
    lookup = FacultyRoomLookup()
    block = lookup.context_block(query)   # None if not applicable
"""

import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "knowledge_base.db"

from pipeline.patterns import COURSE_CODE_RE, FULL_COURSE_ID_RE, canonicalize_course_code  # noqa: E402

# Deliberately conservative keyword list, same rationale as prerequisite_
# graph.py's PREREQ_QUERY_RE: a false negative just falls back to plain
# retrieval (today's behavior, unchanged); a false positive injects an
# irrelevant verified-fact block into context.
FACULTY_ROOM_QUERY_RE = re.compile(
    r"who teaches|which instructor|instructor for|teaches the (theory|lab)|"
    r"office room|which room|what room|room number|"
    r"faculty (for|of|teaching)",
    re.IGNORECASE,
)


class FacultyRoomLookup:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def is_faculty_room_query(self, query: str) -> bool:
        return bool(FACULTY_ROOM_QUERY_RE.search(query)) and bool(COURSE_CODE_RE.search(query))

    def _resolve_course_rows(self, code: str, conn: sqlite3.Connection) -> list[tuple]:
        """Returns matching CourseDetails rows for `code`. Exact match first
        (code already includes a section, e.g. CSE101-01); otherwise every
        section under that base code (e.g. CSE101-01, CSE101-02, ...)."""
        cur = conn.cursor()
        cur.execute(
            "SELECT Course, TheoryInitial FROM CourseDetails WHERE Course = ?",
            (code,),
        )
        rows = cur.fetchall()
        if rows:
            return rows
        cur.execute(
            "SELECT Course, TheoryInitial FROM CourseDetails WHERE Course LIKE ?",
            (f"{code}-%",),
        )
        return cur.fetchall()

    def context_block(self, query: str) -> str | None:
        """Verified 'course -> instructor -> office room' text block, or
        None if the query doesn't ask this, the course isn't found, the
        course reference is ambiguous across multiple sections with
        different instructors, or the instructor has no FacultyList entry."""
        if not self.is_faculty_room_query(query):
            return None

        # Prefer an exact section-level ID (e.g. "CSE101-01") when the query
        # gives one -- unambiguous by construction, no need to guess a
        # section. Only fall back to bare base codes (e.g. "CSE101") for
        # course mentions a full ID didn't already account for.
        full_ids = {m.group(0).replace(" ", "").upper() for m in FULL_COURSE_ID_RE.finditer(query)}
        covered_bases = {re.match(r"[A-Za-z]{2,4}\d{3}[A-Za-z]*", fid).group(0) for fid in full_ids}
        base_codes = {
            canonicalize_course_code(m.group(0)) for m in COURSE_CODE_RE.finditer(query)
        } - covered_bases
        codes = full_ids | base_codes
        if not codes:
            return None

        conn = sqlite3.connect(self.db_path)
        blocks = []
        try:
            for code in codes:
                rows = self._resolve_course_rows(code, conn)
                if not rows:
                    continue
                initials = {r[1] for r in rows if r[1]}
                if len(initials) != 1:
                    # Either no instructor on file, or genuinely ambiguous
                    # (different sections, different instructors) -- don't
                    # guess which one the user meant.
                    continue
                initial = initials.pop()
                course_label = rows[0][0] if len(rows) == 1 else code

                cur = conn.cursor()
                cur.execute(
                    "SELECT Name, Room, Designation, Email FROM FacultyList WHERE Initial = ?",
                    (initial,),
                )
                faculty_row = cur.fetchone()
                if not faculty_row:
                    continue
                name, room, designation, email = faculty_row
                blocks.append(
                    f"Verified instructor information for {course_label} (from the "
                    f"CourseDetails and FacultyList knowledge base): theory section "
                    f"taught by {name} ({designation}), office room {room}, email {email}."
                )
        finally:
            conn.close()

        return "\n".join(blocks) if blocks else None
