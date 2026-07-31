"""
Re-mines ONLY the negative-label half of results/conformal_calibration_
labels.csv, using a much stricter filter -- found via manual inspection
(2026-07-29) that the original negative-mining criterion in scripts/build_
conformal_calibration_labels.py was badly flawed: it treated any non
-hedge-phrase-matching sentence from an Out-of-Scope-query answer as a
"confident false claim," but manual review of the actual 305 collected
negatives showed the vast majority were CORRECT, appropriate responses --
refusals, redirections, privacy/ethics declines, clarifying questions
("Not sure what 'it' refers to..."), and correct policy statements ("No,
you are not guaranteed a seat...") -- not hallucinations. HEDGE_RE only
caught a narrow set of exact phrasings ("I don't have enough information");
it missed the much broader, much more common class of appropriately-worded
refusals this project's own dataset construction favors for Out-of-Scope
answers (see calibrate_abstention.py's own docstring: "reference answers
are deliberate refusals, not facts").

Running calibration on that contaminated set would have taught the
conformal mechanism to distrust the system's own CORRECT behavior --
worse than no calibration at all. Caught by manually inspecting the data
before trusting it, per this project's standing "verify before trusting"
discipline.

Fix: a claim is only even a CANDIDATE hallucination if it positively
contains specific, checkable factual content (email/phone/URL/room/
semester+year/day+time/money/course-code -- the same kind of content the
GOOD positive-labeled examples all actually have) AND does not match an
expanded refusal/redirection/opinion/clarifying-question pattern. This is
a positive-evidence filter, not another blocklist attempt -- it accepts
that most Out-of-Scope answers are appropriate refusals and will correctly
find FEW negative examples, rather than mislabeling refusals as
hallucinations to inflate the count.

Does NOT re-run generation (expensive, ~190 Ollama calls) -- reuses the
raw claims already collected in results/conformal_calibration_labels.csv's
negative rows (source=="out_of_scope"), re-filtering them under the
stricter criterion. Since the original script only kept claims that passed
its (flawed) filter, this can only SHRINK the negative set, never add back
claims that were correctly excluded before -- a limitation disclosed here,
not hidden: any genuine hallucination that happened to ALSO match the old
HEDGE_RE (unlikely, since hedges are refusals and hallucinations are
assertions, but not impossible) would already be gone. A full re-run from
scratch would be more thorough but costs another ~190 generations; this
re-filter is the pragmatic middle ground given that constraint.

Usage: python scripts/fix_conformal_negative_labels.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.patterns import COURSE_CODE_RE

RAW_PATH = ROOT / "results" / "conformal_calibration_labels.csv"
OUT_PATH = ROOT / "results" / "conformal_calibration_labels_fixed.csv"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 2026-07-31: fixed the same greedy-trailing-period bug found and fixed
# in scripts/measure_ir_metrics.py's identical pattern -- see that file's
# comment for the full mechanism. Doesn't change this script's already
# concluded finding (conformal calibration needs human labels), just
# keeps the pattern consistent across the codebase.
PHONE_RE = re.compile(r"(\+?\d{1,3}[-\s])?\d{3,5}[-\s]?\d{5,8}")
URL_RE = re.compile(r"https?://\S+|www\.\S+|bracu\.ac\.bd\S*")
SEMESTER_YEAR_RE = re.compile(r"(Spring|Summer|Fall)\s+\d{4}", re.IGNORECASE)
DAY_TIME_RE = re.compile(
    r"\b(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)\b[^.]{0,30}\d{1,2}:\d{2}", re.IGNORECASE)
MONEY_RE = re.compile(r"(Tk\.?|BDT|\$)\s?[\d,]+|[\d,]+\s?(Tk\.?|BDT|taka)", re.IGNORECASE)
ROOM_RE = re.compile(r"\b(room|floor|building|basement)\b[^.]{0,15}[a-zA-Z]?\d", re.IGNORECASE)

# Positive-evidence filter: a claim must contain at least one of these
# specific, checkable fact shapes to even be considered a candidate
# hallucination -- the same kind of content the verified-retrieval
# positive examples actually have (contact numbers, addresses, specific
# policy numbers, course codes, ...). Generic prose, refusals, and opinions
# never match any of these, by design.
SPECIFIC_FACT_PATTERNS = [EMAIL_RE, PHONE_RE, URL_RE, SEMESTER_YEAR_RE, DAY_TIME_RE, MONEY_RE, ROOM_RE, COURSE_CODE_RE]

# Expanded refusal/redirection/opinion/clarifying-question detector --
# broader than the original HEDGE_RE, built directly from the actual
# false-negative examples found in manual review, not guessed in the
# abstract. A claim matching any of these is excluded even if it happens
# to also contain a specific-fact pattern (e.g. "email the Finance Office
# at queries-accounts@bracu.ac.bd" style redirections DO contain a real
# email, but are appropriate redirections, not hallucinated facts about
# the QUESTION asked -- excluded on purpose, a conservative choice that
# shrinks the negative set further rather than risk mislabeling).
REFUSAL_RE = re.compile(
    r"i (can'?t|cannot|won'?t|am not (able|going to)|don'?t have)|"
    r"not something i|that'?s (on your|outside|beyond|unrelated|a (personal|privacy))|"
    r"check with|contact (your|the)|consult (your|the)|inform (your|the)|"
    r"no,? you (are not|cannot|can'?t)|"
    r"is there anything else|"
    r"would (be|serve) (better|you better)|would serve you|"
    r"i'?m not the right|your own department|"
    r"not appropriate|wouldn'?t be appropriate|"
    r"integrity|privacy|impersonat|fabricat|"
    r"outside what (this|the)|beyond what (this|the)|not something this|"
    r"depends on|personal (financial|academic)|"
    r"not sure what|could you tell me|which .* do you mean|"
    r"\?\s*$",
    re.IGNORECASE,
)


def is_specific_fact(claim: str) -> bool:
    return any(p.search(claim) for p in SPECIFIC_FACT_PATTERNS)


def is_refusal_or_opinion(claim: str) -> bool:
    return bool(REFUSAL_RE.search(claim))


def main():
    df = pd.read_csv(RAW_PATH)
    negatives = df[df["source"] == "out_of_scope"].copy()
    positives = df[df["source"] == "verified_retrieval"].copy()
    print(f"Original: {len(negatives)} negative-labeled claims, {len(positives)} positive-labeled claims")

    kept, dropped_no_fact, dropped_refusal = [], 0, 0
    for _, row in negatives.iterrows():
        claim = str(row["claim"])
        if not is_specific_fact(claim):
            dropped_no_fact += 1
            continue
        if is_refusal_or_opinion(claim):
            dropped_refusal += 1
            continue
        kept.append(row)

    fixed_negatives = pd.DataFrame(kept)
    print(f"After stricter filter: {len(fixed_negatives)} negative claims kept "
          f"(dropped {dropped_no_fact} with no specific-fact content, "
          f"{dropped_refusal} that matched refusal/opinion/clarification patterns)")

    if len(fixed_negatives):
        print("\nSurviving negative examples (should look like actual specific-but-ungrounded claims):")
        for _, row in fixed_negatives.iterrows():
            print(f"  - {str(row['claim'])[:150]!r}")

    out = pd.concat([positives, fixed_negatives], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} total labeled claims to {OUT_PATH} "
          f"({(out['is_correct']).sum()} positive, {(~out['is_correct']).sum()} negative)")


if __name__ == "__main__":
    main()
