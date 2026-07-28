"""
Interactive CLI chat with the real, tested system: hybrid BM25+ChromaDB
retrieval (default lambda=0.5, the paper's deployed configuration) feeding
a locally hosted Llama-3.1-8B via Ollama.

Usage:
    python scripts/chat.py                # lambda=0.5 (full hybrid)
    python scripts/chat.py --lambda 1.0    # BM25-only
    python scripts/chat.py --lambda 0.0    # vector-only
    python scripts/chat.py --no-retrieval  # no-retrieval baseline
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hybrid_retriever import HybridRetriever
from pipeline.ollama_client import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5,
                         help="0=vector-only, 1=BM25-only, 0.5=full hybrid (default)")
    parser.add_argument("--no-retrieval", action="store_true",
                         help="Skip retrieval entirely (no-retrieval baseline)")
    parser.add_argument("--show-context", action="store_true",
                         help="Print the retrieved chunks before the answer")
    args = parser.parse_args()

    mode = "no-retrieval" if args.no_retrieval else f"lambda={args.lam}"
    print(f"=== BRAC advising chatbot (local Llama-3.1-8B, {mode}) ===")
    print("Type 'exit' or 'quit' to end.\n")

    retriever = None if args.no_retrieval else HybridRetriever()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        context = None
        if retriever is not None:
            results = retriever.retrieve(query, args.lam)
            context = "\n\n".join(r["text"] for r in results)
            if args.show_context:
                print("\n--- retrieved context ---")
                for r in results:
                    tag = " [exact match]" if r.get("exact_match") else ""
                    print(f"  [{r['metadata'].get('table', '?')}] score={r['score']:.3f}{tag}: {r['text'][:100]!r}")
                print("--- end context ---\n")

        t0 = time.perf_counter()
        answer = generate(query, context)
        dt = time.perf_counter() - t0

        print(f"Assistant ({dt:.1f}s): {answer}\n")


if __name__ == "__main__":
    main()
