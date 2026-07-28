"""
Composes the four novelty components into one retrieval-to-context pipeline:

  1. Adaptive fusion routing (HybridRetriever.retrieve_adaptive) -- routes
     entity-heavy queries to RRF, open-ended queries to the validated linear
     default, instead of one fixed lambda/fusion mode for every query.
  2. Cross-encoder reranking (Reranker) -- rescopes the routed candidate set
     with a joint query-document relevance signal before generation.
  3. Prerequisite-graph augmentation (PrerequisiteGraph) -- for prerequisite-
     chain questions, prepends a graph-verified multi-hop chain to the
     context, which text retrieval alone cannot recover.
  4. Confidence-gated abstention (AbstentionGate) -- if the calibrated
     query_confidence threshold isn't met, decline rather than let the LLM
     generate from weak/no grounding. Overridden by a sufficient-context
     check (see build_context) before the raw confidence signal is trusted.

Each component is independently real and independently ablatable (every one
has its own class, its own module, and was tested standalone before being
wired together here) -- this module only composes them; it introduces no
new retrieval or scoring logic of its own.

Sufficient-context override on the abstention decision
---------------------------------------------------------
Google Research's "Sufficient Context" work (Joren et al., arXiv:2411.06037)
finds that combining a context-sufficiency signal with model/retrieval
confidence beats confidence alone by >10% on selective accuracy-coverage
trade-offs, because confidence signals like top1-top2 margin have blind
spots confidence alone can't see (this project found its own instance of
that blind spot: near-duplicate paraphrase chunks all scoring similarly high
produces a tiny margin despite unambiguous, correct retrieval -- see
hybrid_retriever.py's retrieve() docstring). This project already computes
two cheap, concrete sufficiency signals as a side effect of retrieval
itself, so no extra LLM call is needed to get them:
  - exact_match: the top retrieved chunk matched a course code/alias
    directly against corpus metadata, not by lexical/semantic similarity --
    about as strong a "the corpus actually has this entity" signal as
    exists for this domain.
  - graph_augmented: the prerequisite graph found a verified multi-hop
    chain for the course(s) in the query.
Either one overrides an abstain decision from the confidence-threshold
gate, on the principle that direct structural evidence the corpus contains
the answer should outrank a noisy confidence heuristic that's already known
to misfire on this corpus's duplicate-paraphrase structure.
"""

import difflib
import re
import time
from pathlib import Path

from pipeline.abstention import ABSTENTION_MESSAGE, AbstentionGate
from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import normalize_entities, translate_to_english
from pipeline.prerequisite_graph import PrerequisiteGraph
from pipeline.reranker import Reranker

# Cheap, distinctive Bengali-transliteration ("Banglish") function-word gate:
# only pay the extra translation LLM call for queries that actually look
# code-mixed, not every English query. Grounded in the tRAG/MultiRAG
# cross-lingual retrieval pattern (question-translation before retrieval,
# see literature review 2026-07-27): monolingual dense retrieval is
# "marginally useful or even harmful when questions are posed in low-
# resource languages, since relevant information is likely available only
# in different languages" -- our EnglishQA content on the same topics as a
# Banglish question is exactly this case. Words chosen for being common in
# transliterated Bengali and rare/absent in standard English (avoids
# false-positive-triggering on genuine English queries).
BANGLISH_MARKER_RE = re.compile(
    r"\b(ki|ache|achi|korte|korar|koro|korbo|hobe|hoise|jonno|gula|guli|koto|"
    r"diye|theke|kora|lagbe|dorkar|amar|tumi|apni|kintu|ebong|akhon|sathe|"
    r"niye|ta\b|er\b|naki|thik|jaiga|jaygay)\b",
    re.IGNORECASE,
)


def is_likely_banglish(query: str) -> bool:
    return bool(BANGLISH_MARKER_RE.search(query))


# Cheap gate for entity-normalization retrieval fallback (2026-07-28, see
# NORMALIZE_ENTITIES_PROMPT / normalize_entities): a code review found
# COURSE_CODE_RE-based exact-match fails entirely on non-canonical
# separators like "CSE 220"/"CSE-220" (fixed directly via regex, see
# patterns.py), but a fixed regex still can't catch every non-canonical
# form -- unusual punctuation, or a misspelled/informal faculty name not in
# the alias table. Rather than pay an LLM call on every query, only trigger
# normalization when the retriever's own is_entity_heavy() found nothing
# AND the query still contains a loose letters-near-digits pattern (a
# broader net than COURSE_CODE_RE itself) that suggests a code-like token
# may be present in some form the retriever didn't recognize.
LOOSE_ENTITY_HINT_RE = re.compile(r"[A-Za-z]{2,6}[.\s_-]{0,2}\d{2,4}")


def looks_like_unrecognized_entity(query: str, retriever) -> bool:
    return not retriever.is_entity_heavy(query) and bool(LOOSE_ENTITY_HINT_RE.search(query))


# Misspelled/informal-name gate (2026-07-28): found by direct ablation
# (scripts/eval_entity_normalization.py) that looks_like_unrecognized_entity
# above -- despite normalize_entities()'s own docstring claiming to handle
# "misspelled or informally-written faculty names" -- can never actually
# fire for a pure name query, since LOOSE_ENTITY_HINT_RE requires letters
# followed by DIGITS. "What is Dr. Shatobdo's email?" (misspelling of
# "Shatabda") has no digits anywhere, so normalize_entities() was silently
# never called for exactly the case its own docstring says it covers -- a
# real mismatch between stated and actual scope, not a hypothetical one.
#
# Two narrow, high-precision patterns catch the common phrasings this
# corpus's own test queries use for a faculty lookup: a title (Dr./Mr./...)
# immediately followed by a capitalized word, or a capitalized word
# immediately followed by a possessive "'s". Both are strong, low-false-
# -positive signals that the word is meant as a person's name specifically
# (as opposed to matching on any capitalized word, which would trigger
# constantly on ordinary sentence-initial capitalization). The candidate
# name is only treated as "unrecognized" if it does NOT already appear (as
# a full name or as a distinctive token) in the retriever's own faculty
# indexes -- so a correctly-spelled name never reaches this gate at all,
# same principle as looks_like_unrecognized_entity requiring is_entity_heavy
# to already be False.
TITLE_NAME_RE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+([A-Z][a-zA-Z]{2,})")
POSSESSIVE_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})'s\b")


def _name_candidates(query: str) -> set[str]:
    return ({m.group(1) for m in TITLE_NAME_RE.finditer(query)}
            | {m.group(1) for m in POSSESSIVE_NAME_RE.finditer(query)})


def looks_like_unrecognized_name(query: str, retriever) -> bool:
    candidates = _name_candidates(query)
    if not candidates:
        return False
    known_tokens = retriever.faculty_name_token_index
    known_full_names = " ".join(n for n, _ in retriever.faculty_name_index)
    for candidate in candidates:
        token = candidate.lower()
        if token in known_tokens:
            continue  # already resolvable via the token index -- not this gate's job
        if token in known_full_names:
            continue  # short fragment of some full name string -- let the substring match handle it
        return True
    return False


def fuzzy_correct_name(query: str, retriever, cutoff: float = 0.72) -> str | None:
    """Deterministic fuzzy correction of a likely-misspelled faculty name,
    tried BEFORE the LLM-based normalize_entities() for name candidates
    specifically. Found by direct ablation (scripts/eval_entity_
    normalization.py) that the local 8B LLM's own phonetic spelling
    guesses are unreliable on harder misspellings ("Shatobdo" corrected to
    "Shatordo" -- still wrong; "Kaikobad" to "Kaikabad" -- still wrong),
    while a simple typo it got right ("Kaykobadd" to "Kaykobad"). Faculty
    names are a small, CLOSED set (~223 people) already fully indexed in
    faculty_name_token_index -- a string-distance search over a known,
    bounded vocabulary is a better fit for this specific problem than
    generative correction from a general-purpose LLM with no special
    knowledge of this roster. Returns a corrected query if a confident
    single match is found for every unrecognized candidate, else None (the
    caller falls back to the LLM path, which still helps on genuinely
    novel/unlisted names this deterministic approach can't cover)."""
    candidates = _name_candidates(query)
    known_tokens = list(retriever.faculty_name_token_index.keys())
    corrected_query = query
    found_any = False
    for candidate in candidates:
        token = candidate.lower()
        if token in retriever.faculty_name_token_index:
            continue
        matches = difflib.get_close_matches(token, known_tokens, n=1, cutoff=cutoff)
        if matches:
            corrected_query = re.sub(re.escape(candidate), matches[0].capitalize(), corrected_query, count=1)
            found_any = True
    return corrected_query if found_any else None

# EnglishQA/BanglishQA-sourced chunks are written as "Question: <value>\n
# Answer: <value>" (see scripts/build_corpus.py's INGESTION_PLAN labels) --
# so a retrieved chunk's *source question* is available verbatim in its text,
# not just its similarity score. Checking whether it near-exactly matches the
# user's actual query is a second, independent sufficiency signal alongside
# exact_match_any/has_graph: this corpus is 81.7% near-duplicate-paraphrase
# rows (scripts/analyze_duplicates.py), so "the retrieved chunk IS this exact
# question, just possibly reworded" is common and structurally verifiable,
# rather than inferred from a noisy confidence margin. Confirmed as a real,
# fixable false-abstain case (2026-07-27): "Where is the EEE Department?"
# retrieved its own verbatim Q/A pair (top1_score=0.824) but fell just under
# the calibrated open_ended threshold (0.8388) and was wrongly declined.
# 0.90 matches the same near-duplicate-cluster threshold already validated
# in analyze_duplicates.py, for consistency.
QUESTION_LINE_RE = re.compile(r"^Question(?:\s*\(Banglish\))?:\s*(.+)$", re.MULTILINE)
QUESTION_MATCH_THRESHOLD = 0.90

# Ambiguous-entity handling (2026-07-28): found by direct testing that a
# genuinely ambiguous entity-heavy query -- "What is Rahman's office room?",
# where 16 different real faculty members share that surname -- silently
# retrieved only 5 of the 16 (final_k's default), showed the LLM no signal
# that the query was ambiguous at all, and the LLM then answered with full
# confidence naming just one of them, giving no indication that 15 other
# real people matched too. This is a genuine correctness/safety gap for an
# advising chatbot: a confidently-stated answer that is silently arbitrary
# among many equally-valid candidates is worse than an answer that visibly
# hedges. AMBIGUOUS_ENTITY_MAX bounds how many candidates get pulled into
# context when this fires, so a pathological case (nothing this corpus
# currently has, but not provably impossible) can't blow up the prompt.
AMBIGUOUS_ENTITY_MAX = 20
AMBIGUOUS_ENTITY_NOTICE = (
    "NOTE: This query's named person/entity matches MULTIPLE distinct "
    "records in the knowledge base (listed below). Do not silently pick "
    "one. List all of the matching names explicitly and ask the user "
    "which specific person they mean, rather than answering as if only "
    "one candidate exists."
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _question_match_ratio(query: str, chunk_text: str) -> float:
    m = QUESTION_LINE_RE.search(chunk_text)
    if not m:
        return 0.0
    source_question = m.group(1).strip()
    return difflib.SequenceMatcher(None, _normalize(query), _normalize(source_question)).ratio()


def _zigzag_by_confidence(scored_items: list[tuple[float, str]]) -> list[str]:
    """Reorders (confidence_score, text) pairs into a "sandwich": highest
    confidence first, second-highest last, third-highest second,
    fourth-highest second-to-last, working inward -- see build_context's
    confidence-ordered context assembly docstring for why (Liu et al. 2024,
    "Lost in the Middle": models use information at the start/end of a long
    context far more reliably than the middle). Ties broken by original
    (already score-sorted-descending) order via a stable sort."""
    ordered = sorted(scored_items, key=lambda x: x[0], reverse=True)
    result: list[str] = [None] * len(ordered)  # type: ignore[list-item]
    lo, hi = 0, len(ordered) - 1
    for i, (_, text) in enumerate(ordered):
        if i % 2 == 0:
            result[lo] = text
            lo += 1
        else:
            result[hi] = text
            hi -= 1
    return result


class NovelPipeline:
    def __init__(self, retriever: HybridRetriever = None, reranker: Reranker = None,
                 prereq_graph: PrerequisiteGraph = None, abstention_gate: AbstentionGate = None,
                 lambda_entity: float = 0.9, lambda_open: float = 0.5,
                 rerank_pool_size: int = 10, final_k: int = 5,
                 use_reranker: bool = False, use_graph: bool = True, use_abstention: bool = True,
                 use_query_translation: bool = False, use_entity_normalization: bool = True):
        # use_query_translation defaults to False as of 2026-07-27: tested
        # directly on the Banglish eval set (isolated ablation, same
        # fine-tuned-embeddings + no-reranker setup either way) and found
        # NOT statistically significant either direction (BLEU: 0.810 with
        # vs 0.819 without, p=0.68; all 4 metrics p>0.46). Honest
        # explanation: the embedding fine-tuning already trained directly on
        # (Banglish question, English answer) pairs, so it already bridges
        # the cross-lingual gap on its own -- translation is redundant on
        # top of that, adding a real extra LLM call's latency for no
        # measured benefit. Left implemented and available (pass
        # use_query_translation=True) since it may still help in a genuinely
        # cross-lingual scenario this test set doesn't probe (a Banglish
        # query about content that exists ONLY in EnglishQA, not
        # BanglishQA) -- just not proven to help on the data actually
        # tested, so not justified as the cost-adding default.
        # use_reranker defaults to False as of 2026-07-27: confirmed via a
        # direct ablation that the generic (non-fine-tuned) cross-encoder
        # reranker became a net negative once the domain-fine-tuned
        # embeddings made initial ranking strong -- it was significantly
        # *losing* to full_hybrid on BLEU/METEOR with the reranker on, and
        # tied (no significant difference) with it off. Pass
        # use_reranker=True explicitly to restore the old behavior if ever
        # needed (e.g. if the reranker is later fine-tuned too, which could
        # flip this back).
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or (Reranker() if use_reranker else None)
        self.prereq_graph = prereq_graph or (PrerequisiteGraph() if use_graph else None)
        if use_abstention and abstention_gate is None:
            abstention_gate = AbstentionGate()  # raises if not yet calibrated
        self.abstention_gate = abstention_gate
        self.lambda_entity = lambda_entity
        self.lambda_open = lambda_open
        self.rerank_pool_size = rerank_pool_size
        self.final_k = final_k
        self.use_query_translation = use_query_translation
        # use_entity_normalization defaults to True as of 2026-07-28,
        # confirmed by direct ablation (scripts/eval_entity_normalization.py,
        # n=8 deliberately-malformed queries verified against
        # knowledge_base.db): course-code-separator variants (double space,
        # underscore, period between letters and digits) go from 0/3 hits to
        # 3/3 once normalization is on. Misspelled-name coverage
        # (looks_like_unrecognized_name) was added the same day after this
        # same ablation found the ORIGINAL gate (looks_like_unrecognized_
        # entity) could structurally never fire for a pure-name query -- see
        # that function's docstring for why -- despite this feature's own
        # motivation explicitly citing misspelled names as a target case.
        self.use_entity_normalization = use_entity_normalization

    def build_context(self, query: str) -> tuple[str | None, dict]:
        """Returns (context_string_or_None, meta). meta always has: route
        ("entity_heavy"/"open_ended"), fusion, lambda, abstain (bool),
        graph_augmented (bool), query_confidence, retrieval_s, rerank_s."""
        t0 = time.perf_counter()
        pool_size = self.rerank_pool_size if self.reranker else self.final_k
        results, route_meta = self.retriever.retrieve_adaptive(
            query, self.lambda_entity, self.lambda_open, top_n=pool_size)

        # Ambiguous-entity widening (see AMBIGUOUS_ENTITY_MAX docstring
        # above): if this entity-heavy query's exact-match resolves to more
        # candidates than the current pool holds, re-retrieve with a wider
        # top_n sized to fit all of them (capped), so no true candidate is
        # silently dropped purely because of the usual pool-size cutoff --
        # before this fix, a query like "What is Rahman's office room?"
        # (16 real matches) only ever retrieved 5, arbitrarily.
        exact_ids = self.retriever.exact_match_ids(query) if route_meta["route"] == "entity_heavy" else set()
        is_ambiguous_entity = len(exact_ids) > 1
        final_k_for_query = self.final_k
        if is_ambiguous_entity and len(exact_ids) > pool_size:
            wider_n = min(len(exact_ids), AMBIGUOUS_ENTITY_MAX)
            results, route_meta = self.retriever.retrieve_adaptive(
                query, self.lambda_entity, self.lambda_open, top_n=wider_n)
            # Propagate the widened size through every downstream truncation
            # step (translation/normalization union, final_results) too --
            # otherwise the widened pool would just get cut back down to
            # the usual final_k a few lines later, undoing this entirely.
            pool_size = wider_n
            final_k_for_query = wider_n

        # Cross-lingual query-translation retrieval (tRAG pattern, see module
        # docstring): a Banglish query's relevant content may only be well-
        # indexed in the corpus's English-heavy portion (EnglishQA is ~2.2x
        # larger than BanglishQA), so a monolingual Banglish-only search can
        # structurally miss it. Only triggered for queries that look
        # code-mixed (cheap word-list gate, no LLM cost for English queries),
        # and only ADDS candidates -- never removes any the original query
        # already found -- so this cannot make a query worse than before,
        # only give it a second, English-routed chance to find relevant
        # content the original phrasing didn't surface.
        translated_query = None
        if self.use_query_translation and is_likely_banglish(query):
            try:
                translated_query = translate_to_english(query)
            except Exception:
                translated_query = None  # translation failure -> fall back silently to original-only
            if translated_query and translated_query.strip().lower() != query.strip().lower():
                translated_results, _ = self.retriever.retrieve_adaptive(
                    translated_query, self.lambda_entity, self.lambda_open, top_n=pool_size)
                by_doc_id = {d["doc_id"]: d for d in results}
                for d in translated_results:
                    existing = by_doc_id.get(d["doc_id"])
                    if existing is None or d["score"] > existing["score"]:
                        by_doc_id[d["doc_id"]] = d
                results = sorted(by_doc_id.values(), key=lambda d: d["score"], reverse=True)[:pool_size]

        # Entity-normalization retrieval fallback (see looks_like_
        # unrecognized_entity/normalize_entities docstrings): only pays the
        # LLM call when the retriever's own exact-match logic found nothing
        # AND the query still contains a loose entity-shaped hint --
        # same dual-query union pattern as translation above, so this can
        # only ever add candidates, never remove ones the original query
        # already found.
        normalized_query = None
        if self.use_entity_normalization and looks_like_unrecognized_name(query, self.retriever):
            # Deterministic fuzzy match against the closed faculty-name
            # vocabulary tried first (see fuzzy_correct_name docstring);
            # only fall back to the LLM's generative guess if no confident
            # match exists in the known roster.
            normalized_query = fuzzy_correct_name(query, self.retriever)
            if normalized_query is None:
                try:
                    normalized_query = normalize_entities(query)
                except Exception:
                    normalized_query = None
        elif self.use_entity_normalization and looks_like_unrecognized_entity(query, self.retriever):
            try:
                normalized_query = normalize_entities(query)
            except Exception:
                normalized_query = None

        # Shared re-retrieval + union step for whichever branch above set
        # normalized_query (fuzzy name correction, LLM name correction, or
        # LLM course-code correction) -- previously nested inside only the
        # elif branch, a real bug found by direct testing: fuzzy_correct_name
        # was correctly rewriting "Shatobdo" to "Shatabda" (confirmed in
        # meta["normalized_query"]) but the corrected retrieval never
        # actually ran, so results never picked up the exact match, and the
        # query still abstained despite the correction being right.
        if normalized_query and normalized_query.strip().lower() != query.strip().lower():
            normalized_results, _ = self.retriever.retrieve_adaptive(
                normalized_query, self.lambda_entity, self.lambda_open, top_n=pool_size)
            by_doc_id = {d["doc_id"]: d for d in results}
            for d in normalized_results:
                existing = by_doc_id.get(d["doc_id"])
                if existing is None or d["score"] > existing["score"]:
                    by_doc_id[d["doc_id"]] = d
            results = sorted(by_doc_id.values(), key=lambda d: d["score"], reverse=True)[:pool_size]
        t1 = time.perf_counter()

        if self.reranker and results:
            final_results = self.reranker.rerank(query, results, top_k=final_k_for_query)
        else:
            final_results = results[:final_k_for_query]
        t2 = time.perf_counter()

        graph_block = self.prereq_graph.context_block(query) if self.prereq_graph else None

        # Sufficient-context signals (exact_match/question_match) are checked
        # over the WIDER pre-rerank pool (`results`, size=rerank_pool_size),
        # not just the post-rerank final_results -- a genuine match can sit
        # at rank 6-10 and get reranked out of the final top-k. Critically,
        # if such a match exists ONLY in the wider pool, it's injected into
        # the context here rather than just used to justify not abstaining:
        # overriding an abstain on evidence the generator never actually
        # sees would be worse than abstaining -- it invites the LLM to guess
        # from context that doesn't contain the answer while the system
        # claims sufficient grounding.
        final_ids = {d["doc_id"] for d in final_results}
        pool_matches = [d for d in results
                        if d["doc_id"] not in final_ids and (d.get("exact_match")
                        or _question_match_ratio(query, d["text"]) >= QUESTION_MATCH_THRESHOLD)]

        # Confidence-ordered context assembly (2026-07-28): previously,
        # context was assembled as [graph_block, pool_matches, final_results]
        # regardless of confidence -- so the single highest-confidence chunk
        # (final_results[0], already the top-ranked candidate) ended up
        # buried in the MIDDLE of the assembled context whenever a graph
        # block or any pool match existed, not at the start where it would
        # be most useful. Liu et al. (2024, TACL, "Lost in the Middle") show
        # LLMs use information at the start/end of a long context far more
        # reliably than information in the middle, independent of whether it
        # technically fits within the context window. We now assign every
        # context piece a confidence score (graph blocks and pool matches --
        # both already-verified structural evidence, exact metadata matches
        # or near-duplicate source questions -- score at least as high as
        # any retrieval-ranked candidate, consistent with how
        # UNAMBIGUOUS_MATCH_SCORE already treats structural evidence as
        # authoritative rather than merely probable) and arrange them in
        # zig-zag order: highest confidence first, second-highest last,
        # third-highest second, working inward -- a "sandwich" that puts the
        # most useful evidence at both ends and the least useful in the
        # middle, rather than truncation-order or retrieval-rank order.
        scored_parts = []
        if is_ambiguous_entity:
            # Tied at float("inf") with graph_block below (there is no float
            # that sorts higher than infinity) -- _zigzag_by_confidence's
            # sort is stable, so appending this FIRST, before graph_block,
            # is what actually guarantees it wins the tie and lands in the
            # highest-confidence slot, not the score value itself. This
            # matters: the notice is only useful if the model reads it
            # before committing to an answer, so it must reliably win that
            # slot whenever ambiguity fires.
            scored_parts.append((float("inf"), AMBIGUOUS_ENTITY_NOTICE))
        if graph_block:
            scored_parts.append((float("inf"), graph_block))  # verified structural fact, not a probabilistic score
        for d in pool_matches:
            scored_parts.append((1e6, d["text"]))  # already-verified sufficiency match (exact_match or near-duplicate question), ranks above any ordinary retrieval score
        for d in final_results:
            scored_parts.append((d["score"], d["text"]))
        context_parts = _zigzag_by_confidence(scored_parts)
        context = "\n\n".join(context_parts) if context_parts else None

        confidence = final_results[0]["query_confidence"] if final_results else 0.0
        top1_score = final_results[0]["query_top1_score"] if final_results else 0.0
        signals = {"query_confidence": confidence, "query_top1_score": top1_score}
        raw_abstain = (self.abstention_gate.should_abstain(signals, route_meta["route"])
                       if self.abstention_gate else False)

        # Sufficient-context override (see module docstring): direct
        # structural evidence the corpus has this entity beats a confidence
        # heuristic with a known blind spot on this corpus. Checks every
        # chunk actually assembled into context (final_results + any
        # pool_matches injected above), not just the rank-1 one -- confirmed
        # as a real bug (2026-07-27): "Who is the theory coordinator for
        # CSE310?" retrieved CSE310's own entry (with the correct answer,
        # ADR) at rank 2, behind CSE350's entry at rank 1, so exact_match on
        # rank-1 alone was False even though the context the model actually
        # receives does contain a genuine exact match.
        context_checked = final_results + pool_matches
        exact_match_any = any(d.get("exact_match") for d in context_checked)
        has_graph = graph_block is not None
        question_match_any = any(
            _question_match_ratio(query, d["text"]) >= QUESTION_MATCH_THRESHOLD
            for d in context_checked
        )
        sufficient_context = exact_match_any or has_graph or question_match_any
        abstain = raw_abstain and not sufficient_context

        meta = {
            **route_meta,
            "abstain": abstain,
            "raw_abstain": raw_abstain,
            "sufficient_context_override": raw_abstain and sufficient_context,
            "graph_augmented": has_graph,
            "is_ambiguous_entity": is_ambiguous_entity,
            "n_ambiguous_candidates": len(exact_ids),
            "exact_match_any": exact_match_any,
            "question_match_any": question_match_any,
            "query_confidence": confidence,
            "query_top1_score": top1_score,
            "translated_query": translated_query or "",
            "normalized_query": normalized_query or "",
            "retrieval_s": t1 - t0,
            "rerank_s": t2 - t1,
        }
        return context, meta

    def answer(self, query: str, generate_fn) -> tuple[str, dict, str | None, float]:
        """generate_fn(query, context) -> str, i.e. pipeline.ollama_client.generate.
        Returns (answer_text, meta, context_used, generation_s). If the
        abstention gate fires, generate_fn is never called -- meta["abstain"]
        records that this was a mechanism decision, not a generation failure."""
        context, meta = self.build_context(query)
        if meta["abstain"]:
            return ABSTENTION_MESSAGE, meta, context, 0.0
        t0 = time.perf_counter()
        # Use the entity-normalized query for generation when one exists,
        # not just retrieval -- found by direct testing (2026-07-28): a
        # misspelled-name query ("What is Dr. Shatobdo's email address?")
        # correctly retrieved the right person's record (context literally
        # contained "Name: Dr. Swakkhar Shatabda ... Email: swakkhar.
        # shatabda@bracu.ac.bd", exact_match_any=True, not abstained), but
        # the LLM still answered "I do not have enough information" when
        # given the ORIGINAL misspelled query alongside correctly-spelled
        # context -- it couldn't reconcile "Shatobdo" (asked) with
        # "Shatabda" (in context) as the same person. Normalization is a
        # spelling correction of the same question, not a paraphrase or
        # translation, so using it for generation is safe and in fact
        # necessary for the retrieval fix to actually reach the user; this
        # is deliberately NOT extended to translated_query (query
        # translation, see build_context) since that changes language, not
        # just spelling, and no evidence supports doing the same swap there.
        generation_query = meta.get("normalized_query") or query
        answer = generate_fn(generation_query, context)
        t1 = time.perf_counter()
        return answer, meta, context, t1 - t0
