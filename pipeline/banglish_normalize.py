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

import re

# Variant -> canonical root. Grouped by semantic root for readability.
_VARIANT_GROUPS = {
    "kor": ["korte", "korar", "koro", "korbo", "kora", "korchi", "korlam",
            "korish", "korlo", "korish", "korbe", "korle", "koris"],
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
BANGLISH_STOPWORDS = {
    "ki", "ta", "er", "ar", "na", "ei", "oi", "o", "ba", "je", "eta", "ota",
    "amar", "tumi", "apni", "kintu", "ebong", "naki", "abar", "niye", "diye",
    "theke", "jonno",
}


def normalize_banglish_token(token: str) -> str:
    return BANGLISH_NORMALIZE_MAP.get(token, token)
