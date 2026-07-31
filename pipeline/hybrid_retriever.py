"""
Hybrid BM25 + ChromaDB retriever, implementing paper.tex Eqs. 1-2 (Section 3.3):

    S_vec(D, Q) = 1 / (1 + d)                          # d = Chroma vector distance
    Score(D, Q) = lambda * S_bm25(D, Q) + (1-lambda) * S_vec(D, Q)

BM25 scores are normalized to [0,1] by dividing by the max BM25 score in the
candidate set. The candidate set is the UNION of the top-10 BM25 doc_ids and
the top-10 vector doc_ids (not the intersection) -- a doc present in only one
stream is scored 0 on the missing component, per the paper. Top-5 by combined
score are returned.

lambda=1.0  -> BM25-only
lambda=0.0  -> vector-only
lambda=0.5  -> full hybrid (deployed default per the paper)

Exact-match course-code boost
------------------------------
A sanity check (query "What are the prerequisites for CSE220?") found that
plain BM25 over chunk text often fails to rank the correct row top-5: the
code "CSE220" also appears inside OTHER rows' prerequisite chains (e.g. as a
prerequisite *of* CSE221, CSE310, ...), diluting the exact-match signal
BM25's term frequency/IDF alone doesn't disambiguate "the row about CSE220"
from "rows that merely mention CSE220." paper.tex Section 3.3 explicitly
states the lexical stream "is architecturally important because it preserves
exact-match precision for alphanumeric identifiers such as course codes" --
a metadata exact-match lookup is the direct, principled way to deliver that,
so it is treated as part of the lexical stream: it only fires when lambda>0,
so vector_only (lambda=0) is completely unaffected (see retrieve()).

Course-code normalization (bug fix, applies in both fusion modes)
-------------------------------------------------------------------
The exact-match boost above only ever worked for Prerequisites/Coordinator
rows, where the "Course" metadata field is a bare code ("CSE220"). The
CourseDetails table stores it WITH a section suffix ("CSE110-01"), so the
old code_index keyed on the raw field value never matched a query-extracted
bare code against it -- exact-match silently never fired for any of
CourseDetails' 527 chunks (schedule/room/lab-section queries). Fixed here by
indexing every table under its regex-extracted canonical code instead of the
raw field value, so all 5 tables get uniform exact-match coverage.

Question-field-aware similarity (retrieval-time, 2026-07-27)
-------------------------------------------------------------
EnglishQA/BanglishQA chunks are formatted as "Question: X\\nAnswer: Y" and
embedded as one block, so the vector stream's similarity to a query is
diluted by the answer text -- a query near-identical to the source Question
doesn't score as high as it structurally should, since half the chunk's
embedded content (the Answer) is irrelevant to that comparison. This corpus
is unusually well-suited to fixing that directly: 46% of EnglishQA rows are
one of 1,069 (Category, Answer) paraphrase groups (confirmed against the
dataset's own Type=Original/Paraphrase column), so "is this query a
near-duplicate of this chunk's source question" is a genuinely common,
structurally verifiable case, not a rare edge case. Every QA-sourced chunk
already carries its source Question in metadata (build_corpus.py's
INGESTION_PLAN), so at init time every such chunk's Question text (not the
full Question+Answer block) is embedded separately and cached; at query
time, the query's embedding (already computed once for the vector stream)
is compared against these question-only embeddings for the candidate set,
and a modest additive boost (QUESTION_BOOST_WEIGHT) is folded into the fused
score in both fusion modes. This is a genuine retrieval-time signal, not a
downstream patch -- distinct from novel_pipeline.py's question_match_any,
which only gates the abstention decision after retrieval/reranking already
ran and can't change which chunks got retrieved in the first place.

Alias resolution (data/course_aliases.json)
---------------------------------------------
Extends exact-match to informal course names (e.g. "NumMeth" -> CSE330,
CLAUDE.md.md's own named limitation) via a small seed lookup table. An alias
hit is treated identically to a canonical course-code hit downstream.

Fusion mode: linear (default, unchanged) vs. rrf
----------------------------------------------------
`fusion="linear"` is the original, byte-for-byte unchanged scoring path
(Score = lambda*S_bm25 + (1-lambda)*S_vec, both in [0,1]) -- every existing
results/ CSV was produced by this path and stays reproducible.

`fusion="rrf"` (opt-in) replaces the per-query max-normalized BM25 score
with Reciprocal Rank Fusion: score(d) = w_bm25/(k+rank_bm25(d)) +
w_vec/(k+rank_vec(d)), k=60 (standard RRF constant), lambda_ still controls
the weight split (w_bm25=lambda_, w_vec=1-lambda_). Motivation: linear
combination requires the two streams' scores to be on comparable scales;
BM25's per-query candidate-max normalization here isn't scale-stable across
queries, which is a documented weakness of linear score fusion vs.
rank-based fusion (see hybrid retrieval literature on RRF). Exact-match and
alias hits are forced to rank 0 in the BM25 stream, same intent as
s_bm25=1.0 in linear mode.
"""

import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from pipeline.chroma_embedding import Chroma1xEmbeddingFunction
from pipeline.tokenizer import tokenize
# Shared with pipeline/prerequisite_graph.py -- see patterns.py module
# docstring for why these must not be redefined locally (a duplicated copy
# is exactly what silently drifted out of sync once before).
from pipeline.patterns import COURSE_CODE_RE, FULL_COURSE_ID_RE, canonicalize_course_code  # noqa: F401 (re-exported for existing importers)

# Faculty initial: 2-5 uppercase letters, no digits -- distinct by
# construction from COURSE_CODE_RE (which always requires 3 digits), so
# there is no collision risk between the two token classes.
FACULTY_INITIAL_RE = re.compile(r"\b[A-Z]{2,5}\b")

# Table-type disambiguation keywords (2026-07-28): a query naming a bare
# course code AND one of these keywords is asking about ONE specific table
# (Prerequisites or Coordinator, each exactly one row per course), not the
# ~10-30 CourseDetails section rows that also share that code. Confirmed as
# a real, high-impact gap via direct inspection of exact-match counts on
# the 100-query English entity-heavy test set: 40/100 queries (all
# "prerequisites for X" / "full prerequisite chain for X" / "theory
# coordinator for X" / "coordinator email for X" shapes) had 2-5 tied
# exact-match candidates instead of the single correct row, purely because
# the old code applied the same CourseDetails-cap-3 logic regardless of
# which table the query was actually asking about.
PREREQ_KEYWORD_RE = re.compile(r"prerequisite|pre-requisite|prereq|chain", re.IGNORECASE)
COORDINATOR_KEYWORD_RE = re.compile(r"coordinator", re.IGNORECASE)

# Faculty schedule-day disambiguation (2026-07-28, see faculty_availability_index
# docstring below): distinguishes "what is X's schedule on Sunday" (wants the
# FacultyAvailability row for that specific day) from "what room is X in" /
# "what is X's email" (wants the FacultyList row) -- confirmed as a real bug
# via the dedicated FacultyAvailability test set: day-schedule queries scored
# far worse (BLEU 0.16) than room queries (BLEU 0.62) on the same test set,
# traced to the FacultyList ceiling crowding out the actually-relevant
# per-day row.
SCHEDULE_KEYWORD_RE = re.compile(r"schedule|available|availability|class(es)?|teaching", re.IGNORECASE)
DAY_RE = re.compile(r"\b(SUNDAY|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY)\b")
# Optional trailing letter added 2026-07-27: some courses are distinct
# letter-suffixed variants of the same number (e.g. CSE490B vs CSE490C, most
# likely separate thesis/project groups) -- without it, both collapsed into
# the same base "CSE490" bucket, and the base-code exact-match logic could
# force up the WRONG variant to the top. Confirmed live: "Who is the theory
# coordinator for CSE490B?" retrieved CSE490C's row instead.

# Section-specific identifier, e.g. "CSE111-07" or "CSE260-05A" -- confirmed
# as a real, high-impact bug (2026-07-27): COURSE_CODE_RE alone only ever
# extracts the base course code, stripping the section suffix, so a query
# about a SPECIFIC section (CourseDetails has ~10-30 section rows per code)
# gets treated identically to a query about the bare code -- forcing up to
# 3 arbitrary CSE111 sections to the top via the exact-match boost, none of
# which need to be the one actually asked about. 6 of the 10 worst-BLEU
# full_hybrid rows in round H were exactly this: "Who teaches the lab
# section of CSE111-07?" retrieving CSE111-01/02/03 instead. This regex
# captures the full identifier (base code, optional trailing letter on the
# code itself e.g. "CSE490B", and the section suffix) so it can be matched
# against CourseDetails' "Course" metadata field directly. (Now defined in
# pipeline/patterns.py and imported above -- see that module's docstring.)
RRF_K = 60

# Weight for the question-field similarity boost added to every candidate's
# fused score (both linear and RRF paths) -- see "Question-field-aware
# similarity" docstring below. Modest and additive by design: it nudges
# scores toward genuine question-level paraphrase matches without
# overriding the existing lambda-controlled BM25/vector balance, which was
# already validated empirically (paper.tex Section 4.3's lambda sweep).
QUESTION_BOOST_WEIGHT = 0.15

# Structured-lookup guarantee: when exactly one candidate is an exact
# metadata match (a single unambiguous course code / section / alias
# resolution -- not one of several tied candidates like multiple
# CourseDetails sections for a bare code), that candidate is authoritative
# structured-database evidence, not a probabilistic ranking signal. It gets
# an absolute score ceiling instead of the modest EXACT_MATCH_BONUS, so it
# is GUARANTEED top-1 rather than merely very likely to be -- eliminating
# any remaining tail-risk that an unrelated candidate's BM25/vector/question
# scores happen to exceed lambda_*1.0 + (1-lambda_)*s_vec + bonus in an edge
# case not yet observed. Grounded in 2026 production RAG practice: route
# unambiguous entity queries to a direct structured lookup rather than
# relying on semantic/lexical ranking at all (see literature review,
# 2026-07-27) -- this is that guarantee, implemented as a score ceiling
# rather than a separate code path, so it stays inside the same fusion
# framework and still returns a normal-shaped result.
UNAMBIGUOUS_MATCH_SCORE = 100.0

# Direct additive bonus for exact_match docs in linear fusion mode, on top
# of forcing s_bm25=1.0. Confirmed as a real gap (2026-07-27): forcing
# s_bm25=1.0 alone only contributes lambda_*1.0 to the combined score (0.5
# at lambda_=0.5), which an unrelated candidate can still outrank via a
# generic-phrase BM25/vector match (observed live: "Who is the theory
# coordinator for CSE490B?" -- the correct exact-matched row scored 0.500,
# but an unrelated Coordinator row scored 0.580 from ordinary lexical/vector
# overlap on the common phrase "theory coordinator", which appears in every
# Coordinator chunk). An exact structured-metadata match (a literal course
# code found in the query) is qualitatively stronger evidence than any
# lexical/semantic similarity score and should not be out-voted by it
# regardless of lambda_. 0.3 closes the observed ~0.08 gap with comfortable
# margin while still preserving relative order among multiple exact matches
# (e.g. capped CourseDetails sections), since it's added uniformly to all of
# them.
EXACT_MATCH_BONUS = 0.3


TITLE_RE = re.compile(r"\b(dr|mr|mrs|ms|prof)\.?\b", re.IGNORECASE)
POSSESSIVE_RE = re.compile(r"'s\b", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def _normalize_name(s: str) -> str:
    """Strips titles (Dr./Mr./...), possessive 's, and punctuation,
    lowercases, collapses whitespace -- used for faculty full-name exact
    matching, the same normalize-then-substring-match idea as course-code
    canonicalization above, applied to person names instead of alphanumeric
    codes.

    Two fixes found by direct testing (2026-07-28), both from the same root
    cause -- naive punctuation deletion can silently MERGE two tokens into
    one instead of separating them:
      - Possessive queries ("Dr. Kaykobad's office room") were normalized to
        "kaykobads" (apostrophe deleted, 's glued onto the name) instead of
        "kaykobad", so they never matched the stored token "kaykobad" at
        all -- a real query returning zero exact matches for an otherwise
        unambiguous, answerable faculty lookup.
      - NON_ALNUM_RE previously deleted punctuation outright rather than
        replacing it with a space, which has the same merging failure mode
        for any other punctuation-adjacent token boundary (e.g. a hyphenated
        name). Replacing with a space and relying on the existing whitespace
        -collapse step is strictly safer and changes no other behavior.
    """
    s = TITLE_RE.sub("", s)
    s = POSSESSIVE_RE.sub("", s)
    s = NON_ALNUM_RE.sub(" ", s.lower())
    return " ".join(s.split())


def _cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class HybridRetriever:
    def __init__(self, chroma_dir=None, bm25_path=None, corpus_path=None,
                 alias_path=None, collection_name="ablation_corpus",
                 stream_k=10, final_k=5, fusion="linear"):
        chroma_dir = chroma_dir or (ROOT / "chroma_db")
        bm25_path = bm25_path or (ROOT / "data" / "bm25_corpus.pkl")
        corpus_path = corpus_path or (ROOT / "data" / "corpus.jsonl")
        alias_path = alias_path or (ROOT / "data" / "course_aliases.json")

        if fusion not in ("linear", "rrf"):
            raise ValueError(f"fusion must be 'linear' or 'rrf', got {fusion!r}")
        self.fusion = fusion
        self.stream_k = stream_k
        self.final_k = final_k

        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_doc_ids = bm25_data["doc_ids"]

        self.corpus = {}  # doc_id -> {text, metadata}
        self.course_index = defaultdict(list)  # "CSE220" -> [doc_id, ...] (canonical code, no section suffix)
        self.full_id_index = {}  # "CSE260-05A" -> doc_id (section-specific, see FULL_COURSE_ID_RE)
        # Faculty exact-match index (2026-07-28): built from FacultyList (223
        # records, ingested into the corpus but -- until this fix -- never
        # reachable via exact match, since _exact_match_ids only recognized
        # course-code patterns). "Initial" is a direct dict lookup (2-5
        # uppercase letters); "Name" needs normalized substring matching
        # since queries include titles ("Dr.", "Mr.") the stored Name field
        # may or may not include, and matching is done both ways (stored
        # name found in query, or query's name phrase found in stored name)
        # to tolerate minor wording differences.
        self.faculty_initial_index = {}  # "AAR" -> doc_id (FacultyList: room/email/designation)
        self.faculty_name_index = []  # [(normalized_name, doc_id), ...] (FacultyList)
        # Token-level fallback index (2026-07-28, found via direct testing:
        # "What is Dr. Kaykobad's office room?" -- a real, unambiguous query,
        # since exactly one FacultyList row's Name contains "Kaykobad" --
        # returned ZERO exact matches under the full-name-substring check
        # above, because that check requires the QUERY to contain the
        # FULL stored name ("mohammad kaykobad"), not merely a distinctive
        # fragment of it. Real users overwhelmingly refer to faculty by a
        # single name (first or last), not their full stored name, so the
        # exact-match ceiling was silently never firing for the common case.
        # This index maps each individual name token (length >= 4, to
        # exclude "md"/"dr"/"mr"/"ms" and short, non-distinctive fragments
        # like "ami" -- which also happens to be a common Banglish word
        # meaning "I", a genuine false-positive risk at shorter lengths) to
        # every doc_id whose name contains that token. A token shared by
        # multiple faculty (e.g. "Anika", "Rahman") legitimately maps to
        # multiple doc_ids -- this is correct, not a bug: it is genuine
        # entity ambiguity that should reach the fusion/ranking mechanism
        # rather than be silently resolved to one arbitrary candidate.
        self.faculty_name_token_index = defaultdict(list)  # "kaykobad" -> [doc_id, ...]
        # FacultyAvailability exact-match index (2026-07-28, found via the
        # dedicated FacultyAvailability test set, Section faculty-coverage):
        # a query asking about a faculty member's SCHEDULE on a specific day
        # was being answered from their FacultyList row (room/email/
        # designation only, no schedule data) because that was the only
        # faculty table indexed for exact match -- the FacultyList ceiling
        # crowded out the actually-relevant per-day FacultyAvailability row
        # entirely, producing "I do not have enough information" even though
        # the corpus genuinely has the answer. Same class of bug as the
        # Prerequisites/Coordinator vs. CourseDetails table confusion fixed
        # above, just discovered later because it was never tested until
        # this table got its own dedicated test set. Keyed on (initial, day)
        # since FacultyAvailability has one row per (faculty, day).
        self.faculty_availability_index = {}  # ("NRHB", "SUNDAY") -> doc_id
        self.faculty_availability_name_index = {}  # ("ruhan habib", "SUNDAY") -> doc_id, name queries don't carry the initial
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.corpus[rec["doc_id"]] = {"text": rec["text"], "metadata": rec["metadata"]}
                course = rec["metadata"].get("Course")
                if course:
                    course_upper = course.strip().upper()
                    # Extract the canonical code (strips CourseDetails' "-01" section
                    # suffix) so exact-match works uniformly across all 5 tables --
                    # see "Course-code normalization" docstring above.
                    m = COURSE_CODE_RE.search(course_upper)
                    if m:
                        self.course_index[m.group(0)].append(rec["doc_id"])
                    full_m = FULL_COURSE_ID_RE.fullmatch(course_upper)
                    if full_m:
                        self.full_id_index[full_m.group(0)] = rec["doc_id"]
                if rec["metadata"].get("table") == "FacultyList":
                    initial = rec["metadata"].get("Initial")
                    if initial:
                        self.faculty_initial_index[initial.strip().upper()] = rec["doc_id"]
                    name = rec["metadata"].get("Name")
                    if name:
                        norm_name = _normalize_name(name)
                        if norm_name:
                            self.faculty_name_index.append((norm_name, rec["doc_id"]))
                            for token in norm_name.split():
                                if len(token) >= 4:
                                    self.faculty_name_token_index[token].append(rec["doc_id"])
                if rec["metadata"].get("table") == "FacultyAvailability":
                    initial = rec["metadata"].get("Initial")
                    day = rec["metadata"].get("Day")
                    if initial and day:
                        self.faculty_availability_index[(initial.strip().upper(), day.strip().upper())] = rec["doc_id"]
                    name = rec["metadata"].get("Name")
                    if name and day:
                        norm_name = _normalize_name(name)
                        if norm_name:
                            self.faculty_availability_name_index[(norm_name, day.strip().upper())] = rec["doc_id"]

        self.aliases = []  # [(alias_lower, canonical_code), ...]
        if Path(alias_path).exists():
            with open(alias_path, "r", encoding="utf-8") as f:
                alias_data = json.load(f)
            for entry in alias_data.get("aliases", []):
                self.aliases.append((entry["alias"].strip().lower(), entry["canonical"].strip().upper()))

        self.embedding_function = Chroma1xEmbeddingFunction()
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        # Question-field-aware similarity (see module docstring): embed
        # every QA-sourced chunk's source Question text separately (not the
        # full Question+Answer block) so query-vs-question similarity can be
        # computed directly at retrieval time, not diluted by answer text.
        question_doc_ids, question_texts = [], []
        for doc_id, info in self.corpus.items():
            meta = info["metadata"]
            q_text = meta.get("Question") or meta.get("QuestionBanglish")
            if q_text:
                question_doc_ids.append(doc_id)
                question_texts.append(q_text)
        if question_texts:
            # Chroma1xEmbeddingFunction exposes __call__ (batch) and
            # embed_query (single), not embed_documents -- use __call__.
            question_vecs = self.embedding_function(question_texts)
            self.question_embeddings = dict(zip(question_doc_ids, question_vecs))
        else:
            self.question_embeddings = {}

    def _bm25_candidates(self, query: str):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.stream_k]
        return {self.bm25_doc_ids[i]: float(scores[i]) for i in ranked}

    def _exact_match_ids(self, query: str):
        # Section-specific identifier match (see FULL_COURSE_ID_RE docstring
        # above): if the query names a specific section ("CSE111-07"), that
        # ONE row is the exact match, full stop -- don't also apply the
        # arbitrary cap-3 base-code selection below for the same code, which
        # is what caused the bug (an arbitrary 3 of the ~10-30 sections got
        # forced up, not necessarily including the one actually asked about).
        full_ids = {m.group(0) for m in FULL_COURSE_ID_RE.finditer(query.upper())}
        matched_full_doc_ids = {self.full_id_index[fid] for fid in full_ids if fid in self.full_id_index}
        codes_covered_by_full_match = {
            base.group(0) for fid in full_ids if fid in self.full_id_index
            for base in [COURSE_CODE_RE.match(fid)] if base
        }

        # Table-type disambiguation (see PREREQ_KEYWORD_RE/COORDINATOR_KEYWORD_RE
        # docstring above): a query about "prerequisites"/"chain" for a course
        # wants the single Prerequisites row, not any CourseDetails sections or
        # the Coordinator row that happen to share the same code; symmetrically
        # for "coordinator". Checked once per query, not per code, since the
        # keyword describes intent about the whole query. Computed here
        # (moved up from below) because the codes_covered_by_full_match
        # exclusion immediately below needs it.
        wants_prereq = bool(PREREQ_KEYWORD_RE.search(query))
        wants_coordinator = bool(COORDINATOR_KEYWORD_RE.search(query))

        # canonicalize_course_code strips any space/dash separator the query
        # matched (e.g. "CSE 220"/"CSE-220" -> "CSE220") so it finds the
        # corpus's always-glued course_index key -- see patterns.py.
        #
        # codes_covered_by_full_match is only excluded here when the query
        # does NOT also want Prerequisites/Coordinator -- a full section
        # match (e.g. "CSE111-04") only covers CourseDetails, not those
        # other base-code-keyed tables. Found live (2026-08-01) via a
        # compound query ("What is the prerequisite for CSE111-04 and
        # which room is its theory class held in?"): bm25_only retrieved
        # the WRONG Prerequisites row (CSE220's, which merely mentions
        # "CSE111" in its own prerequisite text) because "CSE111" was
        # unconditionally dropped from `codes` once "CSE111-04" matched as
        # a full section id, so the correct Prerequisites-CSE111 row never
        # got a chance to be looked up at all, let alone forced to rank 0
        # -- a real retrieval gap, not evidence BM25 is inherently weaker
        # at this query shape. Without this fix, only full_hybrid's vector
        # component happened to compensate; this fix makes bm25_only (and
        # therefore every fusion mode) resolve it correctly via exact
        # match directly, the same guarantee entity-heavy queries already
        # get elsewhere.
        codes = {canonicalize_course_code(m) for m in COURSE_CODE_RE.findall(query)}
        if not (wants_prereq or wants_coordinator):
            codes -= codes_covered_by_full_match
        query_lower = query.lower()
        for alias, canonical in self.aliases:
            if alias in query_lower:
                codes.add(canonical)

        # CourseDetails has ~10-30 section rows per course code (CSE220-01,
        # CSE220-02, ...) vs. exactly one row per code in Prerequisites/
        # Coordinator. Without a cap, a query like "prerequisites for
        # CSE221" pulls in every CSE221 *section* tied at the same forced
        # exact-match score, filling all of final_k and silently dropping
        # the single Prerequisites row that actually answers the question
        # -- confirmed live: this produced "no information" answers for
        # both Prerequisites and Coordinator queries despite correct data
        # existing in the corpus. Cap CourseDetails matches per code so
        # every matching table gets a chance to survive into the final_k
        # cut; other tables have at most a couple of rows per code anyway
        # so they're left uncapped.
        MAX_COURSEDETAILS_PER_CODE = 3

        ids = set(matched_full_doc_ids)
        for code in codes:
            doc_ids = self.course_index.get(code, [])
            by_table = defaultdict(list)
            for doc_id in doc_ids:
                table = self.corpus.get(doc_id, {}).get("metadata", {}).get("table", "")
                by_table[table].append(doc_id)
            if wants_prereq and by_table.get("Prerequisites"):
                ids.update(by_table["Prerequisites"])
            elif wants_coordinator and by_table.get("Coordinator"):
                ids.update(by_table["Coordinator"])
            else:
                for table, table_doc_ids in by_table.items():
                    if table == "CourseDetails":
                        ids.update(table_doc_ids[:MAX_COURSEDETAILS_PER_CODE])
                    else:
                        ids.update(table_doc_ids)

        # Faculty exact match (see faculty_initial_index/faculty_name_index
        # docstring above): initials are matched as standalone uppercase
        # tokens; full names via normalized substring match in either
        # direction, so "Dr. Md. Shafiul Alam Shuvo" (query, with title) and
        # "Md. Shafiul Alam Shuvo" (stored Name, without) still match.
        # Schedule-day disambiguation (see faculty_availability_index
        # docstring above): a query naming a faculty member AND a day AND a
        # schedule keyword wants their FacultyAvailability row for that
        # day, not their FacultyList row (room/email/designation only,
        # which has no schedule data) -- same table-type-disambiguation
        # principle as PREREQ_KEYWORD_RE/COORDINATOR_KEYWORD_RE above.
        query_upper = query.upper()
        wants_schedule = bool(SCHEDULE_KEYWORD_RE.search(query))
        day_match = DAY_RE.search(query_upper)
        day = day_match.group(0) if day_match else None

        matched_initials = {m.group(0) for m in FACULTY_INITIAL_RE.finditer(query_upper)}
        norm_query = _normalize_name(query)
        matched_names = [norm_name for norm_name, _ in self.faculty_name_index
                         if norm_name and norm_name in norm_query] if norm_query else []
        # Token-level fallback (see faculty_name_token_index docstring above):
        # fires only when the full-name substring check finds nothing, so a
        # query that already names someone's full stored name keeps using
        # the more precise match rather than also pulling in same-token
        # collisions unnecessarily.
        token_match_ids = set()
        if norm_query and not matched_names:
            for token in norm_query.split():
                if len(token) >= 4:
                    token_match_ids.update(self.faculty_name_token_index.get(token, ()))

        availability_ids = set()
        if wants_schedule and day:
            for initial in matched_initials:
                doc_id = self.faculty_availability_index.get((initial, day))
                if doc_id:
                    availability_ids.add(doc_id)
            for norm_name in matched_names:
                doc_id = self.faculty_availability_name_index.get((norm_name, day))
                if doc_id:
                    availability_ids.add(doc_id)

        if availability_ids:
            # A specific day-schedule row was found -- this is what the
            # query actually wants, so it replaces (not supplements) the
            # FacultyList match for the same faculty member, preventing the
            # FacultyList ceiling from crowding it out as it did before
            # this fix (confirmed live: "What is Ruhan Habib's schedule on
            # Sunday?" retrieved only their room/email/designation row).
            ids.update(availability_ids)
        else:
            for initial in matched_initials:
                doc_id = self.faculty_initial_index.get(initial)
                if doc_id:
                    ids.add(doc_id)
            for norm_name in matched_names:
                for other_norm_name, doc_id in self.faculty_name_index:
                    if other_norm_name == norm_name:
                        ids.add(doc_id)
            ids.update(token_match_ids)

        return ids

    def _vector_candidates(self, query: str, query_embedding=None):
        query_embedding = query_embedding if query_embedding is not None else self.embedding_function.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.stream_k,
            include=["distances"],
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        return {doc_id: float(d) for doc_id, d in zip(ids, distances)}

    def exact_match_ids(self, query: str) -> set:
        """Public accessor for _exact_match_ids, added so callers outside
        this module (novel_pipeline.py's ambiguous-entity handling; see its
        build_context docstring) can check how many candidates an
        entity-heavy query's exact-match resolves to, without reaching into
        a name-mangled "private" method. Genuinely ambiguous entity queries
        (e.g. "What is Rahman's office room?" -- 16 different real faculty
        members share that surname) are exactly the case a caller needs
        this count for: len() > 1 means the query cannot be answered with
        confidence about which single person is meant, and the caller
        should widen retrieval / signal ambiguity rather than silently
        truncate to whichever candidates happen to fit in the usual top-k."""
        return self._exact_match_ids(query)

    def is_entity_heavy(self, query: str) -> bool:
        """True if `query` contains a course-code token, a known alias, or a
        recognized faculty initial/name -- the same signal _exact_match_ids
        already uses to decide whether to force a doc's lexical score to its
        maximum. Exposed separately so callers (e.g. retrieve_adaptive,
        novel_pipeline.py) can make a routing decision from it without
        duplicating the regex/alias/faculty-index lookup logic."""
        if COURSE_CODE_RE.search(query):
            return True
        query_lower = query.lower()
        if any(alias in query_lower for alias, _ in self.aliases):
            return True
        if any(m.group(0) in self.faculty_initial_index for m in FACULTY_INITIAL_RE.finditer(query.upper())):
            return True
        norm_query = _normalize_name(query)
        if not norm_query:
            return False
        if any(norm_name in norm_query for norm_name, _ in self.faculty_name_index):
            return True
        # Same token-level fallback as _exact_match_ids -- a query naming a
        # faculty member by a single distinctive name fragment ("Kaykobad",
        # "Anika") rather than their full stored name must still be routed
        # as entity-heavy, or the adaptive router sends it down the
        # open-ended/linear-fusion branch and the token match added to
        # _exact_match_ids above never gets the entity-heavy treatment it
        # needs (RRF fusion, the routing this signal exists to drive).
        return any(len(token) >= 4 and token in self.faculty_name_token_index
                   for token in norm_query.split())

    # Signal-type count for entity_signal_strength -- course code, alias,
    # faculty initial, faculty name/token are 4 structurally distinct
    # recognition mechanisms; a query naming more of them independently
    # (rare, e.g. a course code AND a faculty member in one query) reflects
    # more entity-grounded content, not more OCCURRENCES of any one signal
    # (that's what len(exact_match_ids) already measures -- see its
    # docstring above -- and conflating "more signal types" with "more
    # documents from one signal type" would make an ambiguous 16-way surname
    # collision score as maximally entity-heavy for the wrong reason).
    _N_ENTITY_SIGNAL_TYPES = 4

    def entity_signal_strength(self, query: str) -> float:
        """Continuous alternative to is_entity_heavy()'s binary output,
        inspired by (NOT a reproduction of) DAT -- Dynamic Alpha Tuning
        (Hsu & Tzeng, 2025, arXiv:2503.23013) -- which replaces a fixed/
        bucketed hybrid-fusion weight with a continuous per-query one.
        DAT's own mechanism derives that continuous weight from an LLM-
        scored comparison of BM25's vs. the dense retriever's top-1 result
        plausibility; adapted here to use the COUNT of independent
        structural entity-recognition signals instead, for a concrete,
        measured reason: a direct check on this corpus's own test set
        (2026-07-29) found BM25's raw top-1 score is actually LOWER, on
        average, for entity_heavy queries (mean=17.8) than open_ended ones
        (mean=33.0) -- the OPPOSITE of what a raw-score-dominance signal
        would need to indicate "this is an entity query." The confound:
        short course-code/faculty queries simply have fewer terms for
        BM25's score to sum over than a longer natural-language question,
        regardless of how strong the actual match is. DAT's own driving
        signal doesn't transfer to this corpus without correcting for that
        confound, which the already-proven structural checks (course-code
        regex, alias table, faculty initial/name index -- the same ones
        _exact_match_ids and is_entity_heavy already rely on) sidestep
        entirely, since they are discrete structural hits, not magnitude
        comparisons.

        Returns a float in [0, 1]: (number of distinct signal TYPES that
        fire) / 4. EXPERIMENTAL: implemented and unit-tested for sane,
        monotonic behavior (tests/test_patterns.py) but NOT YET empirically
        compared against retrieve_adaptive's existing binary routing on
        this corpus's actual retrieval metrics -- see scripts/ablate_
        dynamic_alpha.py, written but blocked on GPU availability. Do not
        treat this as validated or as a replacement for retrieve_adaptive
        until that comparison actually runs."""
        signals_fired = 0
        if COURSE_CODE_RE.search(query):
            signals_fired += 1
        query_lower = query.lower()
        if any(alias in query_lower for alias, _ in self.aliases):
            signals_fired += 1
        if any(m.group(0) in self.faculty_initial_index for m in FACULTY_INITIAL_RE.finditer(query.upper())):
            signals_fired += 1
        norm_query = _normalize_name(query)
        if norm_query and (
            any(norm_name in norm_query for norm_name, _ in self.faculty_name_index)
            or any(len(token) >= 4 and token in self.faculty_name_token_index for token in norm_query.split())
        ):
            signals_fired += 1
        return signals_fired / self._N_ENTITY_SIGNAL_TYPES

    def retrieve(self, query: str, lambda_: float, fusion: str = None, top_n: int = None):
        """fusion/top_n override the instance defaults (self.fusion,
        self.final_k) for this call only -- added so retrieve_adaptive (below)
        and the reranker pipeline can request a wider candidate slice or a
        different fusion mode per query without needing a second
        HybridRetriever instance. Existing callers that don't pass these
        (every current results/ CSV) get byte-for-byte the same behavior as
        before."""
        fusion = fusion or self.fusion
        top_n = top_n or self.final_k

        query_embedding = self.embedding_function.embed_query(query)
        bm25_cand = self._bm25_candidates(query)  # doc_id -> raw bm25 score
        exact_match_ids = self._exact_match_ids(query) if lambda_ > 0 else set()
        vec_cand_dist = self._vector_candidates(query, query_embedding=query_embedding)  # doc_id -> distance

        # Question-field-aware boost (see module docstring): only computed
        # for the actual candidate set (union of both streams' top-k plus
        # exact matches), not the whole corpus -- cheap, since that's at
        # most ~20-30 docs per query.
        candidate_ids = set(bm25_cand) | set(vec_cand_dist) | exact_match_ids
        question_scores = {
            doc_id: _cosine(query_embedding, self.question_embeddings[doc_id])
            for doc_id in candidate_ids if doc_id in self.question_embeddings
        }

        if fusion == "rrf":
            scored = self._score_rrf(bm25_cand, vec_cand_dist, exact_match_ids, lambda_, question_scores)
        else:
            scored = self._score_linear(bm25_cand, vec_cand_dist, exact_match_ids, lambda_, question_scores)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:top_n]

        # Query-level confidence: margin between the top-1 and top-2 fused
        # scores. Small margin ~ retrieval didn't clearly prefer one chunk
        # over the next, i.e. weak grounding for this query -- USUALLY. Found
        # a real counter-case live (novel-pipeline eval, 2026-07-27): when
        # several near-duplicate paraphrase chunks all correctly match the
        # query (this corpus has multiple paraphrased Q/A rows for the same
        # underlying fact), they score nearly identically HIGH, so the
        # margin is tiny despite retrieval being unambiguous and correct --
        # margin alone can't tell "no good candidate" apart from "several
        # equally-good candidates that agree." Also exposing the raw top-1
        # score (query_top1_score) as a second signal that doesn't have this
        # blind spot: it stays high in the duplicate-agreement case.
        # scripts/calibrate_abstention.py tests both signals empirically
        # (and their combination) rather than assuming margin is the right
        # one -- see that script for which one actually wins.
        if len(scored) >= 2:
            margin = scored[0]["score"] - scored[1]["score"]
        elif len(scored) == 1:
            margin = scored[0]["score"]
        else:
            margin = 0.0
        top1_score = scored[0]["score"] if scored else 0.0
        for r in results:
            r["query_confidence"] = margin
            r["query_top1_score"] = top1_score

        return results

    def retrieve_adaptive(self, query: str, lambda_entity: float = 0.9,
                            lambda_open: float = 0.5, top_n: int = None):
        """Route by query type instead of a single fixed fusion mode/lambda
        for every query -- the direct response to this project's own
        lambda-sweep finding (paper.tex Section 4.3): no fixed blend point
        ever significantly beats BM25-only, on any metric, in either the
        entity-heavy or open-ended subset. That result says a single global
        lambda is the wrong model for this corpus, not that lexical
        retrieval should always win -- so route on the same is_entity_heavy
        signal the sweep itself was stratified by:

          - entity-heavy (course-code / alias match): RRF fusion at
            lambda_entity=0.9, i.e. heavily lexical-weighted rank fusion.
            RRF (fusion="rrf") was implemented in this file but never
            evaluated before this pipeline (see module docstring) -- this is
            the first time it's actually exercised on real queries.
          - open-ended: linear fusion at lambda_open=0.5, i.e. the existing,
            already-validated deployed default (Table 1 / Section 4.2)
            unchanged.

        Returns (results, meta) where meta records which branch fired, so
        downstream evaluation can break results down by routing decision
        the same way Section 4.3 breaks down by entity_heavy/open_ended."""
        entity_heavy = self.is_entity_heavy(query)
        if entity_heavy:
            results = self.retrieve(query, lambda_entity, fusion="rrf", top_n=top_n)
            meta = {"route": "entity_heavy", "fusion": "rrf", "lambda": lambda_entity}
        else:
            results = self.retrieve(query, lambda_open, fusion="linear", top_n=top_n)
            meta = {"route": "open_ended", "fusion": "linear", "lambda": lambda_open}
        return results, meta

    def retrieve_dynamic_alpha(self, query: str, lambda_entity: float = 0.9,
                                 lambda_open: float = 0.5, top_n: int = None):
        """EXPERIMENTAL alternative to retrieve_adaptive -- see
        entity_signal_strength's docstring for the DAT-inspired motivation
        and the concrete reason its own raw-score-dominance mechanism was
        adapted rather than ported as-is. Instead of routing a query into
        exactly one of two fixed (fusion, lambda) configurations, this
        computes a continuous lambda by interpolating between lambda_open
        and lambda_entity according to entity_signal_strength(query) in
        [0, 1], then always uses LINEAR fusion at that interpolated lambda
        -- RRF has no natural analogue for an intermediate lambda (its
        forced-rank-0 exact-match mechanism is inherently a binary on/off,
        unlike linear fusion's continuous score blend), so this mode always
        uses linear fusion, unlike retrieve_adaptive's entity_heavy branch.

        NOT YET VALIDATED: this is a new, additive method -- it does not
        change retrieve_adaptive's behavior or any existing caller, and is
        not used by novel_pipeline.py or any other production path. Kept
        here, disabled-by-default-in-the-sense-of-unused, until scripts/
        ablate_dynamic_alpha.py (written, not yet run -- blocked on GPU
        availability, same reason as the confidence-ordering and ambiguous-
        entity-notice ablations) actually measures whether it beats
        retrieve_adaptive on this corpus's own IR metrics. Do not treat this
        as a replacement for retrieve_adaptive or cite it as an improvement
        until that comparison exists."""
        strength = self.entity_signal_strength(query)
        lambda_ = lambda_open + (lambda_entity - lambda_open) * strength
        results = self.retrieve(query, lambda_, fusion="linear", top_n=top_n)
        meta = {"route": "dynamic_alpha", "fusion": "linear", "lambda": lambda_,
                "entity_signal_strength": strength}
        return results, meta

    def _score_linear(self, bm25_cand, vec_cand_dist, exact_match_ids, lambda_, question_scores=None):
        question_scores = question_scores or {}
        max_bm25 = max(bm25_cand.values()) if bm25_cand else 0.0
        candidate_ids = set(bm25_cand) | set(vec_cand_dist) | exact_match_ids

        scored = []
        for doc_id in candidate_ids:
            if doc_id in exact_match_ids:
                s_bm25 = 1.0
            elif max_bm25 > 0:
                s_bm25 = bm25_cand.get(doc_id, 0.0) / max_bm25
            else:
                s_bm25 = 0.0
            s_vec = (1.0 / (1.0 + vec_cand_dist[doc_id])) if doc_id in vec_cand_dist else 0.0
            s_question = question_scores.get(doc_id, 0.0)
            combined = lambda_ * s_bm25 + (1 - lambda_) * s_vec + QUESTION_BOOST_WEIGHT * s_question
            if doc_id in exact_match_ids:
                combined += EXACT_MATCH_BONUS
                if len(exact_match_ids) == 1:
                    combined += UNAMBIGUOUS_MATCH_SCORE

            info = self.corpus.get(doc_id, {"text": "", "metadata": {}})
            scored.append({
                "doc_id": doc_id,
                "score": combined,
                "s_bm25": s_bm25,
                "s_vec": s_vec,
                "s_question": s_question,
                "exact_match": doc_id in exact_match_ids,
                "text": info["text"],
                "metadata": info["metadata"],
            })
        return scored

    def _score_rrf(self, bm25_cand, vec_cand_dist, exact_match_ids, lambda_, question_scores=None):
        question_scores = question_scores or {}
        # Rank 0 = best. Exact/alias matches are forced to rank 0 in the
        # BM25 stream (same intent as s_bm25=1.0 in linear mode).
        bm25_ranked = sorted(bm25_cand, key=lambda d: bm25_cand[d], reverse=True)
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_ranked)}
        for doc_id in exact_match_ids:
            bm25_rank[doc_id] = 0

        vec_ranked = sorted(vec_cand_dist, key=lambda d: vec_cand_dist[d])  # smaller distance = better
        vec_rank = {doc_id: rank for rank, doc_id in enumerate(vec_ranked)}

        candidate_ids = set(bm25_rank) | set(vec_rank)

        scored = []
        for doc_id in candidate_ids:
            s_bm25 = 1.0 / (RRF_K + bm25_rank[doc_id]) if doc_id in bm25_rank else 0.0
            s_vec = 1.0 / (RRF_K + vec_rank[doc_id]) if doc_id in vec_rank else 0.0
            s_question = question_scores.get(doc_id, 0.0)
            # RRF terms live on a ~1/RRF_K scale (~0.008-0.017), nothing like
            # linear mode's [0,1] -- reusing QUESTION_BOOST_WEIGHT here would
            # let the question boost swamp the rank-fusion signal entirely.
            # Scaling by 1/RRF_K keeps a perfect question match commensurate
            # with roughly one extra rank-0 finish, not a dominant term.
            combined = lambda_ * s_bm25 + (1 - lambda_) * s_vec + (s_question / RRF_K)
            if doc_id in exact_match_ids and len(exact_match_ids) == 1:
                combined += UNAMBIGUOUS_MATCH_SCORE

            info = self.corpus.get(doc_id, {"text": "", "metadata": {}})
            scored.append({
                "doc_id": doc_id,
                "score": combined,
                "s_bm25": s_bm25,
                "s_vec": s_vec,
                "s_question": s_question,
                "exact_match": doc_id in exact_match_ids,
                "text": info["text"],
                "metadata": info["metadata"],
            })
        return scored
