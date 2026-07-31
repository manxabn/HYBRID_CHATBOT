"""
Powers up the ambiguous-entity test set (data/test_queries_ambiguous_entity.
csv, built by scripts/build_ambiguous_entity_test.py) from n=55 to n=220
genuinely ambiguous queries -- a direct response to a reviewer-style
critique that the conditioning-vs-flat-notice comparison (scripts/eval_
ambiguous_entity_notice_quality.py) is only a trend at n=55, not
significant, and needs to be "powered up."

The original set used exactly ONE query template per ambiguous name token
("What is X's office room?"). FacultyList has 4 queryable fields beyond the
name itself (Room, Email, Designation, Status) -- this adds 3 more
templates, each a genuinely distinct, natural question a student would
actually ask, all resolvable from the SAME FacultyList row, all subject to
the SAME name-collision ambiguity (ambiguity is a property of the NAME
TOKEN matching multiple people, not of which field is asked about). This is
a legitimate way to grow n -- more real, distinct questions about the same
real ambiguous people -- not padding via repetition or synthetic
duplication of the existing 55 rows.

Ground truth (true_doc_ids/true_names) is carried over unchanged from the
existing ambiguous rows in test_queries_ambiguous_entity.csv: the set of
candidates a name token resolves to does not depend on which field is
asked about.

Usage: python scripts/expand_ambiguous_entity_test.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCE_PATH = ROOT / "data" / "test_queries_ambiguous_entity.csv"
OUT_PATH = ROOT / "data" / "test_queries_ambiguous_entity_expanded.csv"

# label -> (field name in natural question form)
FIELD_TEMPLATES = {
    "office_room": "What is {name}'s office room?",
    "email": "What is {name}'s email address?",
    "designation": "What is {name}'s designation?",
    "status": "What is {name}'s employment status?",
}


def main():
    df = pd.read_csv(SOURCE_PATH)
    ambiguous = df[df["is_ambiguous"]].reset_index(drop=True)
    print(f"Source: {len(df)} total rows, {len(ambiguous)} genuinely ambiguous (is_ambiguous=True)")

    rows = []
    qid = 0
    for _, r in ambiguous.iterrows():
        token = r["name_token"]
        for field_label, template in FIELD_TEMPLATES.items():
            qid += 1
            query_id = f"AMBX{qid:03d}"
            query = template.format(name=token.capitalize())
            rows.append({
                "query_id": query_id,
                "query": query,
                "name_token": token,
                "field_asked": field_label,
                "is_ambiguous": True,
                "n_true_candidates": r["n_true_candidates"],
                "true_doc_ids": r["true_doc_ids"],
                "true_names": r["true_names"],
                "source_query_id": r["query_id"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} expanded ambiguous queries to {OUT_PATH} "
          f"({len(ambiguous)} people x {len(FIELD_TEMPLATES)} field templates)")
    print(out["field_asked"].value_counts())


if __name__ == "__main__":
    main()
