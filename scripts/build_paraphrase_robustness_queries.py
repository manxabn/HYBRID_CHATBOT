"""
Builds the paraphrase-robustness test set: 20 real base queries (10 entity
-heavy, 10 open-ended, sampled from data/test_queries.csv, seed=7) plus 2
hand-written natural paraphrases each, all three variants sharing the same
verified reference_answer (paraphrasing the question doesn't change the
correct answer). Never tested before in this project -- answers the
question "does the deployed chatbot hold up when the same thing is asked
in different words," which was raised directly and is a real, currently
-untested gap distinct from the judge-paraphrase-invariance check (that
tested the JUDGE's consistency scoring paraphrased judge prompts, not the
CHATBOT's answer quality on paraphrased user queries).

Usage: python scripts/build_paraphrase_robustness_queries.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_PATH = ROOT / "data" / "paraphrase_robustness_base.csv"
OUT_PATH = ROOT / "data" / "paraphrase_robustness_queries.csv"

PARAPHRASES = {
    "Q138": [
        "Could you tell me the email address of CSE437's coordinator?",
        "I need to reach the CSE437 coordinator -- what's their email?",
    ],
    "Q127": [
        "Can you list out the entire prerequisite chain leading up to CSE220?",
        "What's the complete chain of prerequisites I'd need before taking CSE220?",
    ],
    "Q179": [
        "Can you tell me who's in charge of the CSE420-13 lab section?",
        "I want to know the instructor for CSE420-13's lab.",
    ],
    "Q192": [
        "Could you tell me FYS's job title?",
        "What position does FYS hold?",
    ],
    "Q150": [
        "What's the email of the person coordinating CSE430?",
        "How can I email the CSE430 coordinator?",
    ],
    "Q116": [
        "What do I need to have completed before taking CSE360?",
        "Which courses are required before enrolling in CSE360?",
    ],
    "Q194": [
        "What is MVH's title at the university?",
        "Can you tell me MVH's designation?",
    ],
    "Q172": [
        "Who's the instructor for the CSE350-06B lab?",
        "I need to know who teaches CSE350-06B's lab section.",
    ],
    "Q187": [
        "What's FEK's official title?",
        "Could you let me know FEK's designation?",
    ],
    "Q123": [
        "Which courses must I take before CSE430?",
        "What do I need to complete first before taking CSE430?",
    ],
    "Q038": [
        "Is it allowed for visitors to come see students at the Savar campus?",
        "If I'm staying at Savar, can I have guests visit me?",
    ],
    "Q027": [
        "What kind of facilities does TARC offer to students?",
        "Could you describe what amenities are available for students at TARC?",
    ],
    "Q079": [
        "How do I get access to the female Prayer Room?",
        "What's the process for female students to use the Prayer Room?",
    ],
    "Q092": [
        "Do I have to pay an application fee, and what's the amount?",
        "Is there a fee for applying, and how much does it cost?",
    ],
    "Q050": [
        "Are there extra charges for lab or design-based programs on top of tuition?",
        "Besides regular tuition, do lab/design programs cost more?",
    ],
    "Q016": [
        "What's the best way to get in touch with the Finance and Accounts Department at BRACU?",
        "How can I reach BRACU's Finance and Accounts office?",
    ],
    "Q094": [
        "Can non-department students take a minor under ENH?",
        "Is a minor with ENH available to students from other departments?",
    ],
    "Q072": [
        "What does the timeline look like for the Joint PhD program with SOAS?",
        "Can you explain how the SOAS Joint PhD program is organized across its duration?",
    ],
    "Q087": [
        "What happens if someone breaks the TARC rules and regulations?",
        "What penalties are there for violating TARC's rules?",
    ],
    "Q023": [
        "As an EEE student doing a Physics minor, do I still have to satisfy the Physics minor's own prerequisite requirements?",
        "If I'm majoring in EEE and minoring in Physics, are the Physics minor's prerequisites still required for me?",
    ],
}


def main():
    base = pd.read_csv(BASE_PATH)
    rows = []
    for _, r in base.iterrows():
        qid = r["query_id"]
        rows.append({
            "query_id": f"{qid}-orig", "base_query_id": qid, "variant": "original",
            "is_entity_heavy": r["is_entity_heavy"], "query": r["query"],
            "reference_answer": r["reference_answer"],
        })
        for i, p in enumerate(PARAPHRASES[qid], start=1):
            rows.append({
                "query_id": f"{qid}-p{i}", "base_query_id": qid, "variant": f"paraphrase{i}",
                "is_entity_heavy": r["is_entity_heavy"], "query": p,
                "reference_answer": r["reference_answer"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows ({len(base)} base queries x 3 variants) to {OUT_PATH}")


if __name__ == "__main__":
    main()
