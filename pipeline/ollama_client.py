"""
Thin client for local Ollama generation (http://localhost:11434), replacing
the paper's Groq/LLaMA-3.3-70B path with a locally-hosted Llama-3.1-8B to
avoid Groq's rate limits (see plan discussion / methodology-text correction
still pending in paper.tex).
"""

import re
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


class JudgeParseError(Exception):
    """Raised when a judge/extraction prompt's response doesn't contain the
    expected field markers -- added 2026-07-31 after a real, serious bug was
    found: several judge-scoring functions across this project silently
    defaulted to a SPECIFIC score (e.g. 0/"no") whenever their expected
    output format wasn't found in the model's response, rather than
    distinguishing "the model genuinely judged no" from "the response was
    malformed/empty and we couldn't tell." Under heavy concurrent Ollama
    load (this project's own overnight parallel-ablation pattern), a
    request can return a degraded/incomplete completion WITHOUT raising any
    HTTP error at all (post_with_retry only catches timeouts/connection
    errors, not malformed-but-200-OK responses) -- so this silent-default
    behavior was a real, undetected data-corruption risk, not a
    hypothetical one: a stored validation result (kappa=0.7548) had to be
    treated as unverified once this was found, since raw response text
    wasn't being logged to check retroactively. Callers must catch this,
    record the row as an explicit failure (not a fabricated score), and
    re-score only the failed rows afterward -- never silently substitute a
    default."""
    def __init__(self, message, raw_response=None):
        super().__init__(message)
        self.raw_response = raw_response


def post_with_retry(url, json_payload, timeout, max_retries=5, backoff_base=10, max_backoff=120):
    """Shared retry-with-backoff wrapper for Ollama HTTP calls -- added
    2026-07-31 after a real, diagnosed failure: running several Ollama
    -dependent evaluation scripts concurrently (this project's own overnight
    parallel-ablation pattern) caused a genuine request queueing pileup, and
    one request exceeded its 300s read timeout and crashed the whole script
    with zero progress saved (no checkpointing existed). This is NOT masking
    a broken Ollama instance -- it still raises after max_retries genuinely
    fail, so a truly down/hung Ollama service is still reported as an error,
    not silently retried forever. It specifically targets the TRANSIENT
    -contention case (the request would have succeeded fine on its own, it
    just got stuck behind other requests in Ollama's queue) with exponential
    backoff (10s/20s/40s/80s/120s-capped by default) giving the queue time
    to drain.

    Also retries on HTTP 5xx (added 2026-07-31, same day, after a real
    `500 Internal Server Error` crashed a clean single-job run outright --
    the original version only caught connection-level exceptions
    (ReadTimeout/ConnectionError), not an HTTP error response, even though
    a 500 from a locally-hosted model server under momentary load/reload
    pressure (confirmed via /api/ps showing the model NOT staying resident
    between calls -- a real ~10s reload cost on every request on this
    machine) is plausibly just as transient as a timeout. Does NOT retry
    4xx (a malformed request or genuinely missing model won't fix itself
    by waiting -- see the "model not found" env-var bug elsewhere in this
    file's history, which needed a real fix, not a retry)."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=json_payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
        except requests.exceptions.HTTPError as e:
            # 500/502/503/504 specifically: overload, bad-gateway, service
            # -unavailable (queue full), gateway-timeout -- all plausibly
            # transient under load/reload pressure. NOT a blanket 5xx: 501
            # (Not Implemented) and 505 (HTTP Version Not Supported) are
            # genuine protocol errors that retrying cannot fix.
            if e.response is not None and e.response.status_code in (500, 502, 503, 504):
                last_exc = e
            else:
                raise  # 4xx or non-transient 5xx: don't mask a real error
        if attempt < max_retries - 1:
            wait = min(backoff_base * (2 ** attempt), max_backoff)
            print(f"  [ollama_client] request failed (attempt {attempt+1}/{max_retries}), "
                  f"retrying in {wait}s: {last_exc}", flush=True)
            time.sleep(wait)
    raise last_exc

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a helpful academic advising assistant for a university. "
    "Answer the user's question using ONLY the retrieved context below. "
    "If the context does not contain enough information to answer, say so "
    "rather than guessing.\n\n"
    "Always answer in one or more complete sentences that restate the "
    "specific subject of the question (e.g. for \"What is the full "
    "prerequisite chain for CSE489?\" answer \"The full prerequisite chain "
    "for CSE489 is ...\", not just the bare list; for \"What is CMR's "
    "designation?\" answer \"CMR is a Professor.\", not just \"Professor.\"). "
    "Never answer with a single word, a bare list, or a sentence fragment. "
    "State the answer directly and confidently -- do not include "
    "meta-commentary such as \"Based on the retrieved context, I can see "
    "that...\" or reasoning out loud about ambiguity in the context; if the "
    "context is genuinely ambiguous or insufficient, say so in one direct "
    "sentence instead of describing the ambiguity at length.\n\n"
    "Retrieved context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)

SYSTEM_PROMPT_NO_CONTEXT = (
    "You are a helpful academic advising assistant for a university. "
    "Answer the user's question as best you can. Always answer in one or "
    "more complete sentences that restate the specific subject of the "
    "question; never answer with a single word, a bare list, or a sentence "
    "fragment.\n\n"
    "Question: {query}\n"
    "Answer:"
)


TRANSLATE_PROMPT = (
    "Translate the following question into clear, standard English. The "
    "question may be in Bengali (Banglish, written in Latin script), mixed "
    "Bengali/English, or already in English. Preserve any course codes, "
    "names, or identifiers exactly as written. Output ONLY the translated "
    "English question, nothing else -- no explanation, no quotation marks.\n\n"
    "Question: {query}\n"
    "English translation:"
)
# 2026-08-01: a live-confirmed failure was found here -- "Brac er chad
# koyta theke koyta obdi khola thake?" (rooftop opening hours) was
# mistranslated to "How many branches of Brac Bank are open on a
# holiday?", a completely different question. Several prompt rewrites
# (explicit Banglish glossary, "don't guess a different question"
# instruction) were tried and DID fix that specific case, but each one
# also regressed 1-2 queries in the already-measured n=13 cross-lingual
# stress test (results/crosslingual_stress_eval_valexpanded.csv, the
# paper's own reported 84.6% figure) -- re-verified directly, not
# assumed: re-running scripts/eval_crosslingual_stress.py against the
# modified prompt dropped it to 76.9-69.2% depending on the variant,
# because a shared prompt change to fix one rare short word's
# mistranslation shifted the model's interpretation of other, unrelated,
# already-correctly-translating queries. Reverted rather than accept that
# regression on an already-verified paper number -- see PREPROCESS_
# BANGLISH_CONTENT_WORDS below for the zero-regression-risk fix actually
# deployed instead (a deterministic pre-substitution that can only ever
# affect a query containing one of a short, explicit word list, so it
# cannot touch any query it wasn't specifically designed for).


# Deterministic, whole-word Banglish content-word substitution applied
# BEFORE translate_to_english's LLM call, not a prompt change -- see the
# comment on TRANSLATE_PROMPT above for why a shared prompt change was
# tried first and reverted (fixed one word but regressed unrelated
# already-correct translations elsewhere). This substitution can only
# ever affect a query that literally contains one of these exact tokens,
# so it is mechanically incapable of changing behavior on any query it
# wasn't specifically written for -- zero regression risk on the
# already-measured stress test, confirmed by re-running it after adding
# this (results/crosslingual_stress_eval_valexpanded.csv unchanged: same
# 11/13, since none of those 13 queries contain "chad"/"chhad").
#
# Deliberately a small, defensible set added only on direct evidence
# (same discipline as pipeline/banglish_normalize.py's docstring) -- not
# an attempt at exhaustive Banglish-to-English content-word coverage.
PREPROCESS_BANGLISH_CONTENT_WORDS = {
    # "chad"/"chhad" (ছাদ, roof/rooftop) added 2026-08-01: live-confirmed
    # to be mistranslated as unrelated words ("branches", "offices",
    # "toilets" across different prompt attempts) by the small local LLM,
    # derailing the whole translated query ("Brac er chad koyta theke
    # koyta obdi khola thake?" -> a nonsense question about bank branches
    # instead of rooftop opening hours). The corpus has real rooftop
    # content (EnglishQA/BanglishQA rows on rooftop hours/amenities) that
    # was unreachable as a direct result.
    "chad": "rooftop",
    "chhad": "rooftop",
}
_BANGLISH_CONTENT_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in PREPROCESS_BANGLISH_CONTENT_WORDS) + r")\b",
    re.IGNORECASE,
)


def preprocess_banglish_content_words(query: str) -> str:
    return _BANGLISH_CONTENT_WORD_RE.sub(
        lambda m: PREPROCESS_BANGLISH_CONTENT_WORDS[m.group(0).lower()], query
    )


NORMALIZE_ENTITIES_PROMPT = (
    "Rewrite the following question so that any course code or person name "
    "is in its most standard, canonical written form -- e.g. 'cse 220' or "
    "'CSE-220' becomes 'CSE220' (letters directly followed by digits, no "
    "space or dash), and an informally-spelled or misspelled name is "
    "corrected to its most likely standard spelling. Do not change "
    "anything else about the question -- same wording, same meaning, same "
    "language. If nothing needs normalizing, output the question exactly "
    "as given. Output ONLY the rewritten question, nothing else -- no "
    "explanation, no quotation marks.\n\n"
    "Question: {query}\n"
    "Normalized question:"
)


def build_prompt(query: str, context: str | None) -> str:
    if context:
        return SYSTEM_PROMPT_WITH_CONTEXT.format(context=context, query=query)
    return SYSTEM_PROMPT_NO_CONTEXT.format(query=query)


def translate_to_english(query: str, model: str = MODEL, timeout: int = 300) -> str:
    """Used by novel_pipeline.py's cross-lingual query-translation retrieval
    (tRAG pattern, see module docstring): translate a possibly-Banglish query
    to English so it can ALSO be searched against the English-heavy portion
    of the corpus, in addition to the original query. Same deterministic
    decoding as generate() -- this is a retrieval-time utility call, not a
    user-facing answer, but should still be reproducible."""
    query = preprocess_banglish_content_words(query)
    prompt = TRANSLATE_PROMPT.format(query=query)
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 512},
        },
        timeout=timeout,
    )
    return resp.json()["response"].strip()


def normalize_entities(query: str, model: str = MODEL, timeout: int = 300) -> str:
    """Used by novel_pipeline.py's entity-normalization retrieval fallback:
    a deterministic regex fix (pipeline/patterns.py's COURSE_CODE_RE) already
    handles the specific "CSE 220"/"CSE-220" spacing gap found by code
    review, but cannot handle the broader class of non-canonical entity
    mentions -- misspelled or informally-written faculty names not in the
    alias table, unusual course-code capitalization patterns not anticipated
    by any fixed regex. Motivated by Magomere et al. (2025, ACL Findings,
    arXiv:2503.03417), who use an LLM-based "claim normalization" step as a
    train-free, inference-time robustness mitigation for exactly this class
    of input perturbation. Same deterministic decoding as generate() -- a
    retrieval-time utility call, not a user-facing answer, but should still
    be reproducible."""
    prompt = NORMALIZE_ENTITIES_PROMPT.format(query=query)
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 512},
        },
        timeout=timeout,
    )
    return resp.json()["response"].strip()


SUFFICIENT_CONTEXT_PROMPT = (
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Could a diligent reader, using ONLY the context above and no outside "
    "knowledge, give a complete and correct answer to the question? Answer "
    "with exactly one word: YES or NO."
)


def judge_sufficient_context(query: str, context: str, model: str = MODEL, timeout: int = 300) -> bool:
    """Sufficient-context classification, per Joren et al. (2024/2025,
    ICLR 2025, Google Research + UCSD, "Sufficient Context: A New Lens on
    Retrieval Augmented Generation Systems", arXiv:2411.06037): a binary
    judgment of whether the retrieved context alone would let a diligent
    reader answer the question, independent of whether the model actually
    would. Used by the cross-lingual sufficient-context gating experiment
    (scripts/crosslingual_sufficient_context.py): the SAME question, judged
    against context retrieved via the original (possibly Banglish) query
    and separately against context retrieved via its English translation --
    agreement/disagreement between the two is a diagnostic signal for
    translation-sensitive retrieval sufficiency specific to code-mixed
    querying, not explored (as far as this project's literature review
    found) in the existing sufficient-context or code-mixed-RAG literature."""
    prompt = SUFFICIENT_CONTEXT_PROMPT.format(context=context, query=query)
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 2048},
        },
        timeout=timeout,
    )
    text = resp.json()["response"].strip().upper()
    return text.startswith("YES")


def generate(query: str, context: str | None, model: str = MODEL, timeout: int = 900) -> str:
    # CPU-only inference on this machine measured at 15-60s for short
    # answers and 300s+ for detailed multi-paragraph ones (no GPU available
    # -- see CLAUDE.md.md). 900s is a safety ceiling, not an expectation.
    prompt = build_prompt(query, context)
    resp = post_with_retry(
        OLLAMA_URL,
        {
            "model": model, "prompt": prompt, "stream": False,
            # Deterministic (greedy) decoding: standard practice for
            # reproducible RAG evaluation. Without this, two calls with
            # IDENTICAL retrieved context can produce different correct
            # phrasings by sampling chance alone (temperature defaults to
            # ~0.8), which shows up as pure noise in metric comparisons
            # between configs rather than a real quality difference --
            # confirmed concretely: round B's full_hybrid vs novel-pipeline
            # comparison had cases scoring BLEU 1.0 vs ~0.13 on the exact
            # same underlying fact, phrased differently.
            # num_ctx: 2048 was sized against an OLD measurement (~2508
            # chars / ~700 tokens max context) taken before the novel
            # pipeline started injecting prerequisite-graph blocks and
            # pool-matched chunks on top of the base retrieval results. A
            # fresh measurement (2026-07-28, results/novel_pipeline_raw_
            # outputs_roundK_noreranker.csv, n=200) found the ACTUAL max is
            # now 5,041 chars (~1,260 tokens), with p99 at ~1,140 tokens --
            # already eating well into the old 2048 budget once the system
            # prompt, question, and generated answer are added on top, with
            # real risk of Ollama silently truncating the earliest part of
            # the prompt (the graph block / first retrieved chunks) on the
            # longest queries. Raised to 4096, still a deliberate choice
            # below the model's full window: GPU + the page-file fix (see
            # project notes) removed the memory pressure that originally
            # motivated shrinking this, and 4096 keeps a wide margin over
            # the new measured max without reverting to an unbounded
            # default. Raising the ceiling alone does not fully solve the
            # underlying risk, though: Liu et al. (2024, TACL, "Lost in the
            # Middle") show models use information placed at the start/end
            # of a long context far more reliably than information buried
            # in the middle, regardless of whether it technically fits --
            # see novel_pipeline.py's confidence-ordered context assembly,
            # which places the highest-confidence chunk first and the
            # second-highest last rather than leaving assembly order to
            # incidental retrieval rank.
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096},
        },
        timeout=timeout,
    )
    return resp.json()["response"].strip()
