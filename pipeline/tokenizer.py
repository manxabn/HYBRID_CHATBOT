"""
Shared BM25 tokenizer for both index-build time (scripts/build_bm25_index.py)
and query time (pipeline/hybrid_retriever.py) -- must stay identical between
the two or BM25 scores become meaningless.

Lowercased \\w+ tokens, English stopwords removed, NO stemming (stemming
would collapse alphanumeric course codes together and undermine BM25's
exact-match purpose in this design). Stopword removal is necessary: without
it, common query words like "what"/"are"/"the"/"for" summed across a natural-
language question dominate BM25's score over the single rare, high-IDF token
that actually matters (e.g. "cse220"), which was confirmed empirically -- see
plan verification step 2.
"""

import re

import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

from pipeline.banglish_normalize import BANGLISH_STOPWORDS, normalize_banglish_token

TOKEN_RE = re.compile(r"\w+")
STOPWORDS = set(stopwords.words("english")) | BANGLISH_STOPWORDS


def tokenize(text: str):
    # Banglish variant normalization (see banglish_normalize.py) applied
    # before stopword filtering, so e.g. "korte"/"korbo" both collapse to
    # "kor" and then survive filtering as the same, more-discriminative
    # token -- same intent as English stopword removal, just for spelling
    # variants rather than function words.
    tokens = (normalize_banglish_token(t) for t in TOKEN_RE.findall(text.lower()))
    return [t for t in tokens if t not in STOPWORDS]
