"""
Cross-Lingual Sufficient-Context Gating experiment (weakness #10's
proposed new idea): applies Joren et al.'s (ICLR 2025, arXiv:2411.06037)
sufficient-context classification BILINGUALLY -- once against context
retrieved via the original (possibly Banglish) query, and once against
context retrieved via its English translation -- and uses agreement or
disagreement between the two judgments as a diagnostic signal specific to
code-mixed retrieval, not explored elsewhere in the sufficient-context or
code-mixed-RAG literature this project's survey found.

Four possible outcomes per query, each meaning something different:
  YES/YES  -- context is sufficient regardless of retrieval language path;
              translation adds nothing new here.
  NO/NO    -- context is insufficient either way; a genuine retrieval gap,
              not a language-routing issue.
  NO/YES   -- "translation-sensitive sufficiency": the query's own-language
              retrieval alone was NOT enough, but retrieving via the
              English translation found genuinely sufficient context. This
              is the novel, specific failure/success mode this experiment
              is designed to surface -- direct mechanistic evidence for
              WHEN cross-lingual query translation is actually earning its
              keep, not just an aggregate accuracy number.
  YES/NO   -- original-language retrieval was already sufficient; the
              translated-query retrieval path found something the judge
              considers less sufficient (e.g. a plausible-looking but
              off-target match) -- worth inspecting as a possible harm case
              for translation, symmetric to the NO/YES case above.

Run on the 9-query cross-lingual stress test set (Banglish-phrased queries
about EnglishQA-only content, scripts/build_crosslingual_stress_test.py):
this is precisely the set already shown (scripts/eval_crosslingual_stress.py)
to have a large retrieval-accuracy gap between translated and untranslated
retrieval (33.3% vs 77.8% top-5), so it is the set most likely to actually
exercise the NO/YES case this experiment is designed to detect.

Usage: python scripts/crosslingual_sufficient_context.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import judge_sufficient_context, translate_to_english

QUERIES_PATH = ROOT / "data" / "test_queries_crosslingual_stress.csv"
OUT_PATH = ROOT / "results" / "crosslingual_sufficient_context.csv"
POOL_SIZE = 5


def context_for(retriever, query: str) -> str:
    results, _ = retriever.retrieve_adaptive(query, top_n=POOL_SIZE)
    return "\n\n".join(c["text"] for c in results)


def main():
    df = pd.read_csv(QUERIES_PATH)
    retriever = HybridRetriever()

    rows = []
    outcome_counts = {"YES/YES": 0, "NO/NO": 0, "NO/YES": 0, "YES/NO": 0}
    for _, r in df.iterrows():
        query = r["query"]
        try:
            translated = translate_to_english(query)
        except Exception as e:
            print(f"  [warning] translation failed for {r.get('query_id', query)!r}, skipping translated arm: {e}")
            translated = None

        context_original = context_for(retriever, query)
        sufficient_original = judge_sufficient_context(query, context_original)

        if translated and translated.strip().lower() != query.strip().lower():
            context_translated = context_for(retriever, translated)
            sufficient_translated = judge_sufficient_context(query, context_translated)
        else:
            context_translated, sufficient_translated = context_original, sufficient_original

        outcome = f"{'YES' if sufficient_original else 'NO'}/{'YES' if sufficient_translated else 'NO'}"
        outcome_counts[outcome] += 1
        rows.append({
            "query_id": r["query_id"], "query": query, "translated_query": translated,
            "sufficient_original_retrieval": sufficient_original,
            "sufficient_translated_retrieval": sufficient_translated,
            "outcome": outcome,
        })
        print(f"  {r['query_id']}: {outcome}  (query={query[:60]!r})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}\n")
    print("Outcome distribution:")
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count}/{len(df)}")
    n_translation_sensitive = outcome_counts["NO/YES"]
    print(f"\n'Translation-sensitive sufficiency' (NO/YES) cases: {n_translation_sensitive}/{len(df)} -- "
          f"queries where translation was the difference between insufficient and sufficient context.")


if __name__ == "__main__":
    main()
