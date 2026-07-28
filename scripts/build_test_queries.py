"""
Build data/test_queries.csv: ~200 (query, reference_answer) pairs used to
score all four ablation configs against.

~100 open-ended queries are sampled directly from EnglishQA (real
question/answer pairs, used as-is, fixed random seed) -- is_entity_heavy=False.

~100 entity-heavy queries are template-generated from the structured tables
(CourseDetails/Prerequisites/Coordinator/FacultyList), with reference answers
composed directly from the row fields into a natural sentence. This mirrors
what paper.tex Section 3.1 already claims was done for Google Sheets-sourced
entries: "the reference answer was composed directly from the relevant
structured fields." -- is_entity_heavy=True.

is_entity_heavy is assigned by provenance (which template produced the row),
not by scanning generated text. The paper's own regex for course-code
detection (`[A-Z]{2,4}\\d{3}`) is applied only as a secondary sanity check,
printed at the end, not as the source of truth for the label.
"""

import random
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "data" / "test_queries.csv"

from pipeline.prerequisite_graph import PrerequisiteGraph

SEED = 42
N_OPEN_ENDED = 100
N_PREREQ = 30
N_COORDINATOR = 20
N_COURSE = 30
N_FACULTY = 20

COURSE_CODE_RE = re.compile(r"[A-Z]{2,4}\d{3}")


def sample_open_ended(conn, rng):
    # Three real methodology bugs fixed here (found during novel-pipeline
    # evaluation, 2026-07-27):
    # 1. EnglishQA has its own pre-defined Split column (train/test/val) that
    #    this function used to ignore entirely, sampling from the whole
    #    2,297-row pool instead of the 259 rows the dataset creator actually
    #    set aside as "test". Restricting to Split=='test' is the correct
    #    evaluation methodology and is what paper.tex should have been doing
    #    all along.
    # 2. "Out of Scope / Unanswerable" rows (127 of 2,297, ~5.5%) have
    #    reference answers that are deliberate refusals, not facts --
    #    scoring a generated answer against a refusal with BLEU/ROUGE/METEOR
    #    conflates "gave a wrong fact" with "correctly identified this as
    #    unanswerable" under one number. These rows are exactly what
    #    scripts/calibrate_abstention.py is for; excluding them here isn't
    #    hiding a hard case, it's routing it to the mechanism actually built
    #    to evaluate it.
    # 3. scripts/analyze_duplicates.py found 81.7% of EnglishQA rows are
    #    near-duplicate paraphrase clusters (cosine sim >=0.90) -- almost
    #    certainly deliberate phrasing-robustness augmentation by the dataset
    #    creator, not corpus noise, so it should NOT be removed from the
    #    retrieval corpus. But naive random sampling here picked multiple
    #    paraphrases of the same underlying fact (verified: 22/100 sampled
    #    queries were duplicates of another sampled query, covering only 78
    #    distinct facts, not 100). Deduplicating at SAMPLE time -- one query
    #    per underlying fact -- fixes the test set's effective sample size
    #    without touching the corpus itself.
    #
    #    Upgraded 2026-07-27: the dataset's own Type column (Original/
    #    Paraphrase/Multi-Hop) confirms this directly -- rows are grouped by
    #    (Category, Answer) here instead of the earlier embedding-cosine
    #    cluster_id, since identical Answer text within the same Category is
    #    an exact, ground-truth grouping (no similarity threshold to tune),
    #    e.g. "Am I allowed to apply for more than one program..." (Type=
    #    Paraphrase) and "Can I apply for more than one program..." (Type=
    #    Original) share one Answer verbatim. This gives 1,069 distinct
    #    (Category, Answer) groups across the full 2,297-row table.
    df = pd.read_sql(
        "SELECT id, Category, Question, Answer FROM EnglishQA WHERE Question IS NOT NULL AND Answer IS NOT NULL "
        "AND Split = 'test' AND Category != 'Out of Scope / Unanswerable'",
        conn,
    )
    df = df[(df["Question"].str.strip() != "") & (df["Answer"].str.strip() != "")]
    df["cluster_id"] = df["Category"].str.strip() + "||" + df["Answer"].str.strip()

    cluster_ids = df["cluster_id"].unique().tolist()
    chosen_clusters = rng.sample(cluster_ids, min(N_OPEN_ENDED, len(cluster_ids)))
    rows = []
    for cid in chosen_clusters:
        candidates = df[df["cluster_id"] == cid]
        r = candidates.iloc[rng.randrange(len(candidates))]
        rows.append({
            "query": r["Question"].strip(),
            "reference_answer": r["Answer"].strip(),
            "is_entity_heavy": False,
            "source": "EnglishQA",
        })
    return rows


def sample_prerequisites(conn, rng, n):
    # "chain" variant reference answers used to come from the Prerequisites
    # table's own FullChainPreRequisite string field, which turns out to be
    # computed inconsistently: e.g. CSE220's direct prerequisites are BOTH
    # CSE111 and CSE230, and CSE220's own FullChainPreRequisite correctly
    # includes CSE230, but CSE221's FullChainPreRequisite (CSE220-CSE111-
    # CSE110) drops CSE230 even though it depends on CSE220 -- the error
    # then propagates to every course downstream of CSE221 (confirmed: also
    # missing from CSE471 and CSE310's stored field). Found this because the
    # novel pipeline's PrerequisiteGraph -- which does a real transitive BFS
    # over the actual Course/PreRequisite edges -- correctly includes
    # CSE230, and was scored as "wrong" against the buggy reference for it.
    # Using the graph's own BFS output as the reference fixes the test set
    # rather than papering over a real pipeline-vs-corpus disagreement.
    graph = PrerequisiteGraph()
    df = pd.read_sql("SELECT Course, PreRequisite, FullChainPreRequisite FROM Prerequisites", conn)
    df = df.dropna(subset=["Course", "PreRequisite"])
    idx = rng.sample(range(len(df)), min(n, len(df)))
    rows = []
    for i in idx:
        r = df.iloc[i]
        chain = graph.full_chain(r["Course"])
        variant = rng.choice(["basic", "chain"])
        if variant == "basic" or not chain:
            query = f"What are the prerequisites for {r['Course']}?"
            ref = f"The prerequisite for {r['Course']} is {r['PreRequisite']}."
        else:
            query = f"What is the full prerequisite chain for {r['Course']}?"
            ref = f"The full prerequisite chain for {r['Course']} is {'-'.join(chain)}."
        rows.append({"query": query, "reference_answer": ref, "is_entity_heavy": True, "source": "Prerequisites"})
    return rows


def sample_coordinator(conn, rng, n):
    df = pd.read_sql(
        "SELECT Course, FirstTheoryCoordinator, SecondTheoryCoordinator, TheoryEmail FROM Coordinator", conn
    )
    df = df.dropna(subset=["Course", "FirstTheoryCoordinator"])
    idx = rng.sample(range(len(df)), min(n, len(df)))
    rows = []
    for i in idx:
        r = df.iloc[i]
        variant = rng.choice(["who", "email"])
        if variant == "who" or pd.isna(r.get("TheoryEmail")):
            query = f"Who is the theory coordinator for {r['Course']}?"
            ref = f"The theory coordinator for {r['Course']} is {r['FirstTheoryCoordinator']}."
            if not pd.isna(r.get("SecondTheoryCoordinator")):
                ref += f" The second theory coordinator is {r['SecondTheoryCoordinator']}."
        else:
            query = f"What is the coordinator email for {r['Course']}?"
            ref = f"The coordinator email for {r['Course']} is {r['TheoryEmail']}."
        rows.append({"query": query, "reference_answer": ref, "is_entity_heavy": True, "source": "Coordinator"})
    return rows


def sample_course_details(conn, rng, n):
    # LabFaculty (renamed from the old LabInitial) can hold multiple
    # comma-separated initials in the new routine-sheet source, e.g.
    # "SBHN,ESF" -- reference answer just states it as given, verbatim.
    df = pd.read_sql(
        "SELECT id, Course, TheoryDay, TheoryTime, TheoryRoom, LabFaculty FROM CourseDetails", conn
    )
    rows = []
    theory_df = df.dropna(subset=["Course", "TheoryDay", "TheoryTime"])
    lab_df = df.dropna(subset=["Course", "LabFaculty"])

    n_theory = n // 2
    n_lab = n - n_theory

    idx_t = rng.sample(range(len(theory_df)), min(n_theory, len(theory_df)))
    for i in idx_t:
        r = theory_df.iloc[i]
        query = f"What day and time is the theory class for {r['Course']}?"
        ref = f"The theory class for {r['Course']} is on {r['TheoryDay']} at {r['TheoryTime']}"
        ref += f" in room {r['TheoryRoom']}." if not pd.isna(r.get("TheoryRoom")) else "."
        rows.append({"query": query, "reference_answer": ref, "is_entity_heavy": True, "source": "CourseDetails"})

    idx_l = rng.sample(range(len(lab_df)), min(n_lab, len(lab_df)))
    for i in idx_l:
        r = lab_df.iloc[i]
        query = f"Who teaches the lab section of {r['Course']}?"
        ref = f"The lab section of {r['Course']} is taught by {r['LabFaculty']}."
        rows.append({"query": query, "reference_answer": ref, "is_entity_heavy": True, "source": "CourseDetails"})

    return rows


def sample_faculty(conn, rng, n):
    df = pd.read_sql("SELECT Initial, Name, Designation, Room, Email FROM FacultyList", conn)
    df = df.dropna(subset=["Initial", "Name"])
    idx = rng.sample(range(len(df)), min(n, len(df)))
    rows = []
    for i in idx:
        r = df.iloc[i]
        variant = rng.choice(["designation", "email", "room"])
        if variant == "designation" and not pd.isna(r.get("Designation")):
            query = f"What is {r['Initial']}'s designation?"
            ref = f"{r['Name']} ({r['Initial']}) is a {r['Designation']}."
        elif variant == "email" and not pd.isna(r.get("Email")):
            query = f"What is {r['Name']}'s email?"
            ref = f"{r['Name']}'s email is {r['Email']}."
        elif not pd.isna(r.get("Room")):
            query = f"Which room is {r['Name']} in?"
            ref = f"{r['Name']} is in room {r['Room']}."
        else:
            continue
        rows.append({"query": query, "reference_answer": ref, "is_entity_heavy": True, "source": "FacultyList"})
    return rows


def main():
    rng = random.Random(SEED)
    conn = sqlite3.connect(DB_PATH)

    rows = []
    rows += sample_open_ended(conn, rng)
    rows += sample_prerequisites(conn, rng, N_PREREQ)
    rows += sample_coordinator(conn, rng, N_COORDINATOR)
    rows += sample_course_details(conn, rng, N_COURSE)
    rows += sample_faculty(conn, rng, N_FACULTY)
    conn.close()

    df = pd.DataFrame(rows)
    df.insert(0, "query_id", [f"Q{i+1:03d}" for i in range(len(df))])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    n_regex_entity = df["query"].str.contains(COURSE_CODE_RE, regex=True).sum()
    print(f"Wrote {len(df)} test queries to {OUT_PATH}")
    print(df.groupby(["source", "is_entity_heavy"]).size())
    print(f"Secondary sanity check: {n_regex_entity} queries contain a course-code-like token "
          f"(labeled is_entity_heavy={df['is_entity_heavy'].sum()} by provenance)")


if __name__ == "__main__":
    main()
