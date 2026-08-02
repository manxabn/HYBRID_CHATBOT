"""
Banglish equivalent of scripts/eval_paraphrase_robustness.py. BanglishQA
has no Type=Original/Paraphrase column (unlike EnglishQA) and only 15
natural duplicate-answer groups exist, all Split='train' -- too small and
leakage-risky to reuse EnglishQA's exact "real labeled pairs" methodology.

Instead, reuses this project's own already-disclosed LLM-rephrase pattern
(BANGLISH_REPHRASE_PROMPT, scripts/build_crosslingual_stress_test.py),
adapted from English->Banglish to Banglish->Banglish: rephrase a held-out
(Split in val/test, never trained on) BanglishQA question into a
DIFFERENT Banglish phrasing at temperature=0, then test whether the
retriever finds the ORIGINAL row's own chunk (not just any chunk with the
right answer -- same leave-out-self discipline as the English test, since
the rephrased query's own near-identical chunk does not exist in the
corpus here, unlike the English case, so this is somewhat less strict,
but doc_id-level checking is kept for consistency and because BanglishQA
duplicate-answer groups do exist).

This is disclosed as LLM-generated stress-test data, not organic user
data -- same caveat this project already applies to its English->Banglish
stress test (paper.tex's own limitation #10, citing Dogruoz et al. 2023
on LLM-generated code-switched text not being equivalent to organic data).

Usage: python scripts/eval_banglish_paraphrase_robustness.py [--sample-size N]
"""

import argparse
import sys
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import OLLAMA_URL, MODEL, post_with_retry

DB_PATH = ROOT / "knowledge_base.db"
OUT_PATH = ROOT / "results" / "banglish_paraphrase_robustness_raw.csv"
TOP_K = 5
SAMPLE_SEED = 42

BANGLISH_REPHRASE_SAME_LANG_PROMPT = (
    "Rewrite the following Banglish (Bengali-English code-mixed, Latin "
    "script) question using DIFFERENT wording and sentence structure, but "
    "keep it in Banglish and preserve the exact same meaning and any "
    "specific entities (course codes, names, numbers) unchanged. Do not "
    "translate it to English. Output ONLY the rewritten Banglish "
    "question, nothing else, no quotes, no explanation.\n\n"
    "Banglish question: {question}"
)


def rephrase(question: str) -> str:
    resp = post_with_retry(
        OLLAMA_URL,
        {"model": MODEL, "prompt": BANGLISH_REPHRASE_SAME_LANG_PROMPT.format(question=question),
         "stream": False, "options": {"temperature": 0.0, "seed": 42, "num_ctx": 512}},
        timeout=300,
    )
    return resp.json()["response"].strip().strip('"')


def main(sample_size: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, Category, QuestionBanglish, AnswerEnglish
        FROM BanglishQA WHERE Split IN ('val', 'test')
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"Held-out BanglishQA rows available: {len(rows)}")

    df = pd.DataFrame(rows, columns=["id", "category", "question", "answer"])
    sample = df.sample(n=min(sample_size, len(df)), random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"Sampling {len(sample)} for this run", flush=True)

    retriever = HybridRetriever()
    out_rows = []
    for i, r in sample.iterrows():
        orig_doc_id = f"BanglishQA-{r['id']}-chunk0"
        para_q = rephrase(r["question"])
        results = retriever.retrieve(para_q, 0.5, top_n=TOP_K)
        top_doc_ids = [d["doc_id"] for d in results]
        found_original_chunk = orig_doc_id in top_doc_ids
        found_any_correct = any(str(r["answer"]).strip() in d["text"] for d in results)
        out_rows.append({
            "id": r["id"], "category": r["category"],
            "orig_question": r["question"], "para_question": para_q,
            "found_original_chunk_top5": found_original_chunk,
            "found_any_correct_answer_top5": found_any_correct,
        })
        print(f"  [{i+1}/{len(sample)}] id={r['id']}: found_original={found_original_chunk}", flush=True)

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_PATH, index=False)
    n = len(out)
    n_orig = out["found_original_chunk_top5"].sum()
    n_any = out["found_any_correct_answer_top5"].sum()
    print(f"\nn={n}")
    print(f"Found ORIGINAL's own chunk in top-5: {n_orig}/{n} = {n_orig/n:.3f}")
    print(f"Found ANY chunk with correct answer in top-5: {n_any}/{n} = {n_any/n:.3f}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=80)
    args = parser.parse_args()
    main(args.sample_size)
