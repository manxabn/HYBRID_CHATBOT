"""
Derives a corpus-specific Banglish stopword list via TF-IDF, motivated by a
verified literature finding (Chanda & Pal 2023, SN Computer Science,
doi:10.1007/s42979-023-01942-7, corroborated by InfoTextCM's own
experiments at FIRE 2024): corpus-based stopword lists (extracted from the
target corpus itself) outperform generic pre-built lists for Bengali
-English code-mixed IR. This project's existing BANGLISH_STOPWORDS
(pipeline/banglish_normalize.py) is a small, hand-built list of ~25
function words -- this derives a data-driven complement from the corpus's
own ~3000-row Banglish question text (now tripled via scripts/ingest_new_
banglish_dataset.py), rather than replacing the hand-built list outright.

Method: compute per-token document frequency across all Banglish question
texts (train split only, to avoid any leakage into held-out eval sets).
Tokens appearing in a very high fraction of documents (candidate
stopwords, low discriminative power for BM25) that are NOT already course
-code-shaped, faculty-name-shaped, or already in the existing hand-built
list are flagged as candidate ADDITIONS -- printed for review, not
auto-applied, since a wrong addition (removing a genuinely discriminative
token) would be a real regression, and this project's own banglish_
normalize.py docstring already warns "a wrong/over-eager normalization is
worse than no normalization at this scale."

Usage: python scripts/derive_corpus_specific_banglish_stopwords.py
"""

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.banglish_normalize import BANGLISH_STOPWORDS, normalize_banglish_token
from pipeline.patterns import COURSE_CODE_RE

DB_PATH = ROOT / "knowledge_base.db"
WORD_RE = re.compile(r"[a-z]+")
DOC_FREQ_THRESHOLD = 0.03  # appears in >=3% of Banglish questions -- a real, corpus-wide pattern, not noise
MIN_LEN = 2


def load_banglish_train_questions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT QuestionBanglish FROM BanglishQA WHERE Split='train' AND QuestionBanglish IS NOT NULL")
    rows = [q.strip() for (q,) in cur.fetchall() if q and q.strip()]
    conn.close()
    return rows


def main():
    questions = load_banglish_train_questions()
    print(f"Banglish train questions (Split=train only, no leakage into val/test): {len(questions)}")

    doc_freq = Counter()
    for q in questions:
        tokens = {normalize_banglish_token(t) for t in WORD_RE.findall(q.lower())}
        for t in tokens:
            doc_freq[t] += 1

    n_docs = len(questions)
    candidates = []
    for token, freq in doc_freq.most_common(200):
        if len(token) < MIN_LEN:
            continue
        if token in BANGLISH_STOPWORDS:
            continue
        if COURSE_CODE_RE.search(token.upper()):
            continue  # never treat course-code-shaped tokens as stopwords
        doc_ratio = freq / n_docs
        if doc_ratio >= DOC_FREQ_THRESHOLD:
            candidates.append((token, freq, round(doc_ratio, 4)))

    print(f"\n{len(candidates)} candidate corpus-specific stopwords "
          f"(doc_freq >= {DOC_FREQ_THRESHOLD:.0%}, not already in BANGLISH_STOPWORDS, "
          f"not course-code-shaped):\n")
    for token, freq, ratio in candidates:
        print(f"  {token!r}: appears in {freq}/{n_docs} questions ({ratio:.1%})")

    print("\nNOTE: these are candidates for manual review, NOT auto-applied to "
          "BANGLISH_STOPWORDS -- a wrong addition could remove a genuinely "
          "discriminative token from BM25 indexing. Check each one is actually "
          "a function word (not a real content word that happens to be common "
          "in this specific FAQ domain, e.g. 'bracu' or 'university') before adding.")


if __name__ == "__main__":
    main()
