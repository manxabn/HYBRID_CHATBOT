"""
Ad-hoc demo: run one representative query per source table through the live
hybrid retriever + Ollama generator, and print retrieval + the generated
answer for each. Not part of the ablation pipeline -- this is for eyeballing
real chatbot behavior across every ingested data source, on demand.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import generate

QUERIES = [
    ("EnglishQA (BRACU_QA_Dataset_FINAL.csv)",
     "Am I allowed to get a refund of my application fee if I decide not to join?"),
    ("BanglishQA (BRACU_Banglish_QA_Only.csv)",
     "Admission application fee ami kivabe pay korbo?"),
    ("CourseDetails (gsheet_course_routine.csv)",
     "What day and time is the theory class for CSE220?"),
    ("FacultyList (gsheet_faculty_roster.csv)",
     "What is Mr. Annajiat Alim Rasel's email?"),
    ("Coordinator (gsheet_coordinators_v2.csv)",
     "Who is the theory coordinator for CSE330?"),
    ("Prerequisites (prerequisite - Sheet1.csv)",
     "What are the prerequisites for CSE221?"),
    ("FacultyAvailability (gsheet_faculty_availability_raw/)",
     "When is AJA available on Monday?"),
]


def main():
    retriever = HybridRetriever(fusion="linear")

    for source_label, query in QUERIES:
        print("=" * 100)
        print(f"SOURCE: {source_label}")
        print(f"QUERY:  {query}")
        print("-" * 100)

        t0 = time.perf_counter()
        results = retriever.retrieve(query, lambda_=0.5)
        t1 = time.perf_counter()

        print(f"Retrieved {len(results)} chunks in {t1 - t0:.3f}s "
              f"(query_confidence={results[0]['query_confidence']:.4f})" if results else "No results retrieved.")
        for i, r in enumerate(results, 1):
            snippet = r["text"].replace("\n", " ")[:140]
            print(f"  [{i}] table={r['metadata'].get('table')} score={r['score']:.4f} "
                  f"exact_match={r['exact_match']} :: {snippet}")

        context = "\n\n".join(r["text"] for r in results)
        t2 = time.perf_counter()
        answer = generate(query, context)
        t3 = time.perf_counter()

        print(f"\nGENERATED ANSWER ({t3 - t2:.1f}s):")
        print(answer)
        print()


if __name__ == "__main__":
    main()
