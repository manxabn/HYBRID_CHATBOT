"""
Directed prerequisite graph, built from the Prerequisites table's own
Course -> PreRequisite edges (NOT from the FullChainPreRequisite string
column, which is inconsistent at the source: e.g. CSE330's PreRequisite is
MAT216, but its stored FullChainPreRequisite is "MAT120-MAT110" -- MAT216
itself is missing from that chain. Rebuilding the transitive closure from
the one-hop edges via graph traversal is self-consistent and doesn't inherit
that data-entry gap).

Multi-hop prerequisite questions ("what do I need before I can take CSE310?")
require chaining several one-hop edges (CSE310 depends on CSE370 depends on
CSE221 depends on CSE220 depends on CSE111 depends on CSE110 -- a 5-hop
chain). Neither BM25 nor vector retrieval can do this: they retrieve the one
chunk that mentions CSE310's immediate prerequisite, not the transitive
closure, and a 500-char chunk has no room for a 5-hop chain even if it were
computed. This module answers that one query shape correctly by graph
traversal, then hands the verified chain to the LLM as extra context (see
pipeline/novel_pipeline.py) so the final answer still goes through the same
generation step as every other config, rather than bypassing it with a
templated string that would need separate reference-answer handling.
"""

import re
import sqlite3
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "knowledge_base.db"

# Shared with pipeline/hybrid_retriever.py -- previously a separate, locally
# defined copy here had silently drifted out of sync with the retriever's
# (missing the letter-suffix fix for courses like CSE490B/CSE490C), a
# dormant bug caught by code review rather than by a failing query, since
# no letter-suffixed course happens to exist in the Prerequisites table
# yet. See pipeline/patterns.py's module docstring.
from pipeline.patterns import COURSE_CODE_RE, canonicalize_course_code  # noqa: F401 (re-exported for existing importers)

# Heuristic for "this question is asking about a prerequisite chain", not
# just mentioning a course code in passing. Deliberately conservative (few
# keywords) -- a false negative just means the query falls through to plain
# retrieval, unchanged from today; a false positive would inject an
# irrelevant graph block into the context, so keep this list short.
PREREQ_QUERY_RE = re.compile(
    r"prerequisite|pre-requisite|prereq|before i (can |)take|need before|required for|take before",
    re.IGNORECASE,
)

# Distinguishes "what is the full/multi-hop CHAIN for X" (graph augmentation
# genuinely adds value: retrieval alone can't chase transitive edges) from a
# plain "what are the prerequisites for X" (asking only the direct, one-hop
# prerequisite -- plain retrieval already answers this correctly, since it's
# exactly what a single corpus chunk states). Confirmed as a real bug: before
# this check existed, context_block() injected the full chain for BOTH
# question shapes, so e.g. "What are the prerequisites for CSE422?" (direct
# answer: "CSE221") got a full 4-hop chain shoved into context and the model
# over-answered with the whole chain instead of the direct prerequisite --
# not factually wrong, just answering a broader question than was asked, and
# scored as "wrong" against the narrower reference.
CHAIN_QUERY_RE = re.compile(r"chain|full prerequisite", re.IGNORECASE)


class PrerequisiteGraph:
    def __init__(self, db_path=None):
        db_path = db_path or DB_PATH
        self.graph = nx.DiGraph()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT Course, PreRequisite FROM Prerequisites")
        rows = cur.fetchall()
        conn.close()

        self.n_edges = 0
        for course, prereq_field in rows:
            if not course or not prereq_field:
                continue
            course_code = self._extract_code(course)
            if not course_code:
                continue
            self.graph.add_node(course_code)
            for prereq_code in self._parse_prereq_field(prereq_field):
                self.graph.add_edge(course_code, prereq_code)
                self.n_edges += 1

    @staticmethod
    def _extract_code(text: str):
        # canonicalize_course_code strips a query's optional space/dash
        # separator ("CSE 220" -> "CSE220") so a graph lookup on a query-
        # extracted code matches the corpus-derived (always-glued) node
        # names built in __init__ -- see patterns.py. A no-op when text
        # already has no separator (the normal case for corpus fields).
        m = COURSE_CODE_RE.search(text.strip().upper())
        return canonicalize_course_code(m.group(0)) if m else None

    @classmethod
    def _parse_prereq_field(cls, field: str):
        # Raw format: "CSE111 (HP),CSE230 (HP)" -- comma-separated codes,
        # each with a parenthetical qualifier (HP = hard prerequisite, SP =
        # soft prerequisite per the source sheet) that we strip since the
        # graph only models "is a prerequisite of", not qualifier strength.
        codes = []
        for part in field.split(","):
            code = cls._extract_code(part)
            if code:
                codes.append(code)
        return codes

    def is_prereq_query(self, query: str) -> bool:
        return bool(PREREQ_QUERY_RE.search(query)) and bool(COURSE_CODE_RE.search(query))

    def full_chain(self, course_code: str) -> list[str]:
        """Transitive closure of prerequisites for course_code, nearest-first,
        via BFS. Returns [] if the course has no known prerequisites or isn't
        in the graph at all (e.g. a 100-level course, or one not covered by
        the Prerequisites table)."""
        course_code = course_code.strip().upper()
        if course_code not in self.graph:
            return []
        seen = {course_code}
        chain = []
        frontier = [course_code]
        while frontier:
            next_frontier = []
            for node in frontier:
                for prereq in self.graph.successors(node):
                    if prereq not in seen:
                        seen.add(prereq)
                        chain.append(prereq)
                        next_frontier.append(prereq)
            frontier = next_frontier
        return chain

    def context_block(self, query: str) -> str | None:
        """If `query` is a prerequisite-chain question about a course code
        this graph has data for, return a verified-chain text block suitable
        for prepending to the LLM's retrieved context. Returns None if the
        query isn't a prereq query, or the course has no graph entry --
        callers should fall back to plain retrieval in that case, not treat
        None as "no prerequisites"."""
        if not self.is_prereq_query(query) or not CHAIN_QUERY_RE.search(query):
            return None
        codes = {self._extract_code(m) for m in COURSE_CODE_RE.findall(query)}
        blocks = []
        for code in codes:
            chain = self.full_chain(code)
            if chain:
                # Hyphen-joined ("A-B-C"), not arrow-joined ("A -> B -> C"):
                # a code-review-driven isolated ablation (2026-07-28, n=12
                # chain-triggering queries) found graph augmentation
                # significantly HURT BLEU (p=0.025) despite not being a real
                # quality regression -- traced to exactly this formatting
                # mismatch. This corpus's own reference answers and its
                # FullChainPreRequisite source field both use the hyphen
                # convention (e.g. "CSE370-CSE221-CSE220-CSE111-CSE230-
                # CSE110"), but the graph block previously stated the same
                # chain with arrows, and the LLM would adopt the graph
                # block's format in its answer -- an identical fact, scored
                # lower purely because BLEU is an n-gram-overlap metric and
                # "A -> B" and "A-B" share no n-grams at all. Matching the
                # corpus's own convention removes this artifact without
                # changing anything the model actually needs to know.
                blocks.append(
                    f"Verified prerequisite chain for {code} (nearest prerequisite "
                    f"first, from the Prerequisites knowledge base): "
                    f"{'-'.join(chain)}."
                )
        return "\n".join(blocks) if blocks else None
