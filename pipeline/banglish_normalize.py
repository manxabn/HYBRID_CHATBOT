"""
Banglish (transliterated Bengali-English code-mixed) normalization layer for
the shared BM25 tokenizer, per the CMIR/FIRE code-mixed IR shared task
pattern (dictionary-based replacement + consistent normalization -- see
literature review, 2026-07-27): "gd -> good" style variant collapsing,
adapted here to Banglish spelling variants of the same root word.

Why this matters: BM25 is exact-token-match. Banglish has no standard
spelling -- "korte", "korar", "koro", "korbo", "kora" are all inflections of
the same root ("to do") that a Banglish speaker might use interchangeably,
but BM25 treats them as five unrelated tokens, each individually rare (low
term frequency, diluted IDF signal) instead of one token with real
discriminative weight. Collapsing known variants to one canonical form before
indexing/querying directly patches this blind spot, the same way English
stopword removal already patches "what/are/the/for" diluting course-code
signal (see tokenizer.py).

MUST be applied identically at both index-build time
(scripts/build_bm25_index.py) and query time (hybrid_retriever.py's
_bm25_candidates, via the shared tokenize() function) -- same requirement as
the base tokenizer, or BM25 scores become meaningless.

This dictionary is a deliberately small, defensible set built from patterns
actually observed in this project's own Banglish test queries and corpus
content (2026-07-27/28), not an attempt at exhaustive Bengali transliteration
coverage -- a wrong/over-eager normalization is worse than no normalization
at this scale, since it can incorrectly merge genuinely distinct words.
"""

# Variant -> canonical root. Grouped by semantic root for readability.
#
# "kore" added to the "kor" group 2026-07-31: found via a corpus-specific
# TF-IDF stopword derivation (scripts/derive_corpus_specific_banglish_
# stopwords.py, run against the tripled BanglishQA train set, 2414
# questions) that "kore" -- a common conjunctive-participle form of "to
# do" ("having done"/"by doing") -- appears in 13.6% of Banglish questions
# but was never collapsing into "kor" alongside every OTHER inflection of
# the same verb, a real gap in this dictionary, not a new word class.
_VARIANT_GROUPS = {
    "kor": ["korte", "korar", "koro", "korbo", "kora", "korchi", "korlam",
            "korish", "korlo", "korbe", "korle", "koris", "kore"],
    "ache": ["ache", "achi", "achen", "asche"],
    "hobe": ["hobe", "hoise", "hoyeche", "hoye", "hoyese"],
    "gula": ["gula", "guli", "gulo"],
    "kobe": ["kobe", "kokhon"],
    "sathe": ["sathe", "shathe"],
    "akhon": ["akhon", "ekhon"],
    "dorkar": ["dorkar", "lagbe", "lagbo"],
}

BANGLISH_NORMALIZE_MAP = {
    variant: canonical
    for canonical, variants in _VARIANT_GROUPS.items()
    for variant in variants
}

# Banglish function words that are extremely common but carry little
# discriminative weight for BM25 -- the direct Banglish analogue of English
# stopword removal (tokenizer.py), for the same documented reason: without
# removing them, they dilute the rare, high-IDF token that actually matters.
#
# Second block added 2026-07-31 via the same corpus-specific TF-IDF
# derivation noted above: each of these crossed a >=3% document-frequency
# threshold across 2414 real Banglish questions and was manually confirmed
# to be a genuine grammatical/auxiliary word (question word, case-marker
# suffix, modal auxiliary, pronoun), NOT a real content word -- e.g. "kor"/
# "ache"/"hobe" are the CANONICAL TARGETS of the verb-variant groups above
# ("to do"/"to be"/"will be"), which is exactly why they become extremely
# frequent and low-content once every spelling variant collapses into them.
# Candidates that turned out to be genuine domain content words (e.g.
# "brac", "student", "course", "semester", "library") were deliberately
# EXCLUDED, not added, despite also crossing the frequency threshold --
# this project's own banglish_normalize.py docstring already warns that a
# wrong/over-eager addition is worse than no addition at this scale.
BANGLISH_STOPWORDS = {
    "ki", "ta", "er", "ar", "na", "ei", "oi", "o", "ba", "je", "eta", "ota",
    "amar", "tumi", "apni", "kintu", "ebong", "naki", "abar", "niye", "diye",
    "theke", "jonno",
    "kor", "ache", "hobe", "hoy", "jay", "koto", "kothay", "kivabe", "der",
    "te", "ke", "ami", "kon", "pabo", "pare",
}


def normalize_banglish_token(token: str) -> str:
    return BANGLISH_NORMALIZE_MAP.get(token, token)
