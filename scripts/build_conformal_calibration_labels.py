"""
Builds a REAL (claim, context, is_correct) calibration set for pipeline/
conformal_abstention.py's calibrate_threshold_from_labels(), without human
annotation and WITHOUT one LLM judging itself -- direct response to the
critique that conformal abstention is "built but disabled" because the only
calibration path documented so far requires human labels that don't exist
yet (results/human_annotation_sample.csv, prepared, still blank).

This does NOT replace that human-labeling task -- the 50 disagreement
-prioritized rows in that file were specifically selected because they're
genuinely hard/ambiguous cases, which is exactly why they need a human and
why nothing here tries to auto-label THEM. What this DOES provide is a
separate, additional calibration set built entirely from STRUCTURAL facts
already in this project's own data, not from any model's semantic judgment
of an answer's content:

  NEGATIVE labels (is_correct=False), from genuinely unanswerable queries:
    EnglishQA/BanglishQA rows labeled Category="Out of Scope / Unanswerable"
    have NO correct answer anywhere in this corpus, by the dataset's own
    construction (not an LLM's opinion). Run each through the real pipeline;
    if the confidence gate does NOT abstain (the case conformal exists to
    catch as a second line of defense) and the generated answer states a
    concrete, non-hedging claim, that claim is incorrect BY CONSTRUCTION --
    no answer could be correct here. Hedging/refusal claims ("I don't have
    enough information", etc.) are excluded, not mislabeled, since a correct
    refusal isn't the failure case this needs to teach the calibration.

  POSITIVE labels (is_correct=True), from verified-retrieval queries:
    novel_pipeline.py's own question_match_any signal (already used
    elsewhere as a sufficient-context override, not new machinery) is a
    deterministic, structural check: the retrieved chunk's source Question
    field is a near-duplicate (>=0.90 sequence-match ratio) of the user's
    actual query. When that fires, we know with high structural confidence
    the retrieved context IS the correct source record, and its Answer
    field IS the reference_answer already sitting in data/test_queries.csv.
    A generated claim is labeled correct only if it has strong deterministic
    lexical overlap with that known-correct reference_answer -- plain word
    -overlap, not any model's semantic judgment, so this stays independent
    of the NLI model being calibrated AND of the generation LLM.

Both label sources are DISCLOSED as structural/lexical proxies, not human
judgment -- weaker evidence than a real annotator, but categorically
different from (and less circular than) using the same or another LLM to
grade the answer's content. Ambiguous cases (hedged Out-of-Scope answers;
low-but-nonzero overlap on verified-retrieval answers) are excluded rather
than guessed, so only reasonably confident labels enter the calibration set.

Usage: python scripts/build_conformal_calibration_labels.py
"""

import re
import sqlite3
import sys
from pathlib import Path

import nltk
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.novel_pipeline import NovelPipeline
from pipeline.ollama_client import generate
from pipeline.conformal_abstention import decompose_claims
from pipeline.tokenizer import STOPWORDS

DB_PATH = ROOT / "knowledge_base.db"
QUERIES_PATH = ROOT / "data" / "test_queries.csv"
OUT_PATH = ROOT / "results" / "conformal_calibration_labels.csv"

# Refusal/hedge phrases this project's own system prompt explicitly asks the
# model to use ("If the context does not contain enough information to
# answer, say so rather than guessing" -- pipeline/ollama_client.py). A
# claim containing one of these is a correct refusal, not a confident false
# claim, and must be excluded from negative labeling, not mislabeled as
# incorrect.
HEDGE_RE = re.compile(
    r"do(es)?n'?t have (enough|sufficient)|not enough information|"
    r"cannot (find|answer|determine)|do(es)?n'?t (know|contain)|"
    r"no information|not (able to|available)|not (provided|specified) in|"
    r"i (do not|don't) have",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def claim_overlap_ratio(claim: str, reference: str) -> float:
    """Fraction of the claim's own content words also present in the known
    -correct reference answer -- deterministic word overlap, not any
    model's semantic judgment. Asymmetric by design: measures whether the
    claim's specific content is grounded in the reference, not whether the
    reference happens to also match the claim."""
    claim_words = _content_words(claim)
    if not claim_words:
        return 0.0
    ref_words = _content_words(reference)
    return len(claim_words & ref_words) / len(claim_words)


def load_out_of_scope_queries():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT Question FROM EnglishQA WHERE Category='Out of Scope / Unanswerable' "
                "AND Question IS NOT NULL")
    rows = [q.strip() for (q,) in cur.fetchall() if q and q.strip()]
    cur.execute("SELECT QuestionBanglish FROM BanglishQA WHERE Category='Out of Scope / Unanswerable' "
                "AND QuestionBanglish IS NOT NULL")
    rows += [q.strip() for (q,) in cur.fetchall() if q and q.strip()]
    conn.close()
    return rows


def main():
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    pipeline = NovelPipeline()  # use_conformal_backoff=False default -- this script only COLLECTS labels

    negative_queries = load_out_of_scope_queries()
    print(f"Out-of-Scope queries available: {len(negative_queries)}")

    labels = []

    n_skipped_abstained = 0
    n_skipped_hedge = 0
    for i, query in enumerate(negative_queries):
        context, meta = pipeline.build_context(query)
        if meta["abstain"]:
            n_skipped_abstained += 1
            continue
        answer = generate(query, context)
        for claim in decompose_claims(answer):
            if HEDGE_RE.search(claim):
                n_skipped_hedge += 1
                continue
            labels.append({"claim": claim, "context": context or "", "is_correct": False,
                            "source": "out_of_scope", "query": query})
        print(f"  [neg {i+1}/{len(negative_queries)}] abstain={meta['abstain']} "
              f"claims_added={sum(1 for l in labels if l['query'] == query)}", flush=True)

    print(f"Negative labels collected: {sum(1 for l in labels if not l['is_correct'])} "
          f"(skipped {n_skipped_abstained} correctly-abstained, {n_skipped_hedge} correct hedges)")

    df = pd.read_csv(QUERIES_PATH)
    open_ended = df[~df["is_entity_heavy"]].reset_index(drop=True)
    n_no_match = n_skipped_abstained_pos = n_low_overlap = 0
    for i, r in open_ended.iterrows():
        query, reference = r["query"], str(r["reference_answer"])
        context, meta = pipeline.build_context(query)
        if not meta.get("question_match_any"):
            n_no_match += 1
            continue
        if meta["abstain"]:
            n_skipped_abstained_pos += 1
            continue
        answer = generate(query, context)
        for claim in decompose_claims(answer):
            if HEDGE_RE.search(claim):
                continue  # a hedge here is just unhelpful, not a labeled example either way
            overlap = claim_overlap_ratio(claim, reference)
            if overlap >= 0.5:
                labels.append({"claim": claim, "context": context or "", "is_correct": True,
                                "source": "verified_retrieval", "query": query})
            else:
                n_low_overlap += 1  # ambiguous -- excluded, not mislabeled
        print(f"  [pos {i+1}/{len(open_ended)}] question_match_any=True", flush=True)

    print(f"Positive labels collected: {sum(1 for l in labels if l['is_correct'])} "
          f"(skipped {n_no_match} no verified-retrieval match, {n_skipped_abstained_pos} correctly-abstained, "
          f"{n_low_overlap} low-overlap/ambiguous)")

    out = pd.DataFrame(labels)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} total labeled claims to {OUT_PATH} "
          f"({(out['is_correct']).sum()} positive, {(~out['is_correct']).sum()} negative)")


if __name__ == "__main__":
    main()
