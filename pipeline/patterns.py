"""
Single source of truth for the entity-recognition regex patterns shared
across retrieval and prerequisite-graph traversal. Extracted 2026-07-28
after a code review found pipeline/prerequisite_graph.py maintaining its
OWN, separately-defined COURSE_CODE_RE that had silently drifted out of
sync with pipeline/hybrid_retriever.py's copy: the retriever's version was
fixed to accept an optional trailing letter (CSE490B vs. CSE490C, a
confirmed real bug where both collapsed into one "CSE490" bucket), but the
graph module's copy was never updated, so it would still mistruncate a
letter-suffixed course code if one were ever added to the Prerequisites
table. This is the same root-cause class as the table-routing confusion
bugs found elsewhere in this project (Prerequisites/Coordinator vs.
CourseDetails; FacultyList vs. FacultyAvailability) -- a single concept
duplicated across modules with no mechanism forcing them to stay in sync.
Importing one shared pattern here makes that drift structurally impossible
rather than merely fixed-for-now; see tests/test_patterns.py for the
regression test that would have caught the original drift.
"""

import re

# Bare course code: 2-4 letters, an optional single space or dash, 3
# digits, optional trailing letter for letter-suffixed course variants
# (e.g. CSE490B vs. CSE490C -- distinct courses, most likely separate
# thesis/project sections, confirmed via a live query that retrieved the
# wrong one before this was added). The optional "[\s-]?" was added
# 2026-07-28 after a code review found "CSE 220" and "CSE-220" both failed
# to match at all (only "CSE220", no separator, worked) -- confirmed live:
# COURSE_CODE_RE.findall() returned [] for both spaced/dashed forms, which
# would zero out the exact-match ceiling entirely and, combined with the
# now-brittle entity-heavy abstention threshold, cause the system to
# always decline a perfectly answerable query solely because of how the
# student happened to space the course code.
COURSE_CODE_RE = re.compile(r"[A-Za-z]{2,4}[\s-]?\d{3}[A-Za-z]?")

# Section-specific identifier, e.g. "CSE111-07" or "CSE260-05A" -- captures
# the full identifier (base code, optional trailing letter on the code
# itself, and the section suffix) so it can be matched against
# CourseDetails' "Course" metadata field directly, distinct from a bare
# course-code match which only identifies the course, not the section.
FULL_COURSE_ID_RE = re.compile(r"[A-Za-z]{2,4}\d{3}[A-Za-z]*-\d+[A-Za-z]*")


def canonicalize_course_code(raw_match: str) -> str:
    """COURSE_CODE_RE's match may contain a space or dash separator (e.g.
    "CSE 220", "CSE-220") that the corpus's own course-code metadata never
    does (always the glued form, "CSE220") -- callers must canonicalize a
    query-extracted match through this before using it as a course_index
    lookup key, or a legitimately-recognized but differently-spaced code
    will silently fail to find its corpus entry. Safe to call on an
    already-glued code too (no-op)."""
    return raw_match.replace(" ", "").replace("-", "").upper()
