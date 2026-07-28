"""
Retrieval-only evaluation of query translation on the cross-lingual stress
test (data/test_queries_crosslingual_stress.csv, built by
scripts/build_crosslingual_stress_test.py): Banglish-phrased questions whose
answer exists ONLY in EnglishQA, not BanglishQA -- the scenario query
translation was designed for for, unlike the main Banglish test set (whose
null result, paper.tex Section 4.7.3, was explained by the test set
structurally being unable to probe genuine cross-lingual divergence).

Method: retrieval-only (top-5 accuracy via reference_answer substring
match, same convention as lambda_sweep_banglish.py, valid here since these
reference answers ARE the corpus's own verbatim EnglishQA Answer field),
comparing plain adaptive retrieval against the same query with translation
enabled (query translated to English via the LLM, dual-query candidate
fusion) -- exactly NovelPipeline's use_query_translation logic, reimplemented
here directly over the retriever so this stays a cheap retrieval-only check
with no generation cost.

This is a small set (9 queries, all the EnglishQA-test-split rows with no
BanglishQA match once deduplicated by unique answer) -- reported as a
targeted stress test, not a high-powered benchmark; treat accuracy deltas
as suggestive, not significance-tested.

Usage: python scripts/eval_crosslingual_stress.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import translate_to_english

QUERIES_PATH = ROOT / "data" / "test_queries_crosslingual_stress.csv"
OUT_PATH = ROOT / "results" / "crosslingual_stress_eval.csv"
POOL_SIZE = 10


def retrieve_with_translation(retriever, query):
    results, _ = retriever.retrieve_adaptive(query, top_n=POOL_SIZE)
    try:
        translated = translate_to_english(query)
    except Exception:
        translated = None
    if translated and translated.strip().lower() != query.strip().lower():
        translated_results, _ = retriever.retrieve_adaptive(translated, top_n=POOL_SIZE)
        by_doc_id = {d["doc_id"]: d for d in results}
        for d in translated_results:
            existing = by_doc_id.get(d["doc_id"])
            if existing is None or d["score"] > existing["score"]:
                by_doc_id[d["doc_id"]] = d
        results = sorted(by_doc_id.values(), key=lambda d: d["score"], reverse=True)[:POOL_SIZE]
    return results, translated


def main(queries_path: Path, out_path: Path):
    df = pd.read_csv(queries_path)
    retriever = HybridRetriever()

    rows = []
    n_top5_plain, n_top5_translated = 0, 0
    for _, r in df.iterrows():
        query, ref = r["query"], str(r["reference_answer"]).strip()

        plain_results, _ = retriever.retrieve_adaptive(query, top_n=POOL_SIZE)
        plain_hit = any(ref in c["text"] for c in plain_results[:5])

        trans_results, translated_query = retrieve_with_translation(retriever, query)
        trans_hit = any(ref in c["text"] for c in trans_results[:5])

        n_top5_plain += plain_hit
        n_top5_translated += trans_hit
        rows.append({
            "query_id": r["query_id"], "query": query, "translated_query": translated_query,
            "plain_top5_hit": plain_hit, "translated_top5_hit": trans_hit,
        })
        print(f"  {r['query_id']}: plain={plain_hit} translated={trans_hit} "
              f"(translated_query={translated_query!r})", flush=True)

    n = len(df)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Plain top-5 accuracy:      {n_top5_plain}/{n} = {n_top5_plain/n:.3f}")
    print(f"Translated top-5 accuracy: {n_top5_translated}/{n} = {n_top5_translated/n:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    main(args.queries, args.out)
