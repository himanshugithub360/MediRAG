"""Measure retrieval quality so changes can be compared with numbers instead of guesses.

For each question in eval/questions.json, a retrieved passage counts as relevant when it
contains *all* of that question's keywords (case-insensitive). Two scores are reported:

- Hit@K: share of questions with at least one relevant passage in the top K.
- MRR:   mean reciprocal rank of the first relevant passage (1.0 = always ranked first).

Keyword matching is a rough, automatic judge, so treat the numbers as a comparison between
setups, not as an absolute accuracy figure. No LLM calls are made, so this costs nothing.

Usage:
    python eval_retrieval.py              # compare all setups
    python eval_retrieval.py --verbose    # also show the top result per question
"""
import argparse
import json
import os
import time

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from src.config import FINAL_K, INDEX_NAME, LEGACY_NAMESPACE, NAMESPACE, PROJECT_ROOT
from src.helper import download_hugging_face_embeddings
from src.retrieval import build_retriever

QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "eval", "questions.json")


def is_relevant(doc, keywords):
    text = doc.page_content.lower()
    return all(k.lower() in text for k in keywords)


def evaluate(name, retriever, questions, k, verbose):
    hits, rr, start = 0, 0.0, time.time()
    for q in questions:
        docs = retriever.invoke(q["question"])[:k]
        rank = next((i + 1 for i, d in enumerate(docs) if is_relevant(d, q["keywords"])), None)
        if rank:
            hits += 1
            rr += 1 / rank
        if verbose:
            top = docs[0] if docs else None
            where = f"{top.metadata.get('source', '?')} p.{top.metadata.get('page', '?')}" if top else "-"
            print(f"  [{name}] {'rank ' + str(rank) if rank else 'MISS  '} | {q['question']} -> {where}")
    n = len(questions)
    return {"setup": name, "hit": hits / n, "mrr": rr / n, "secs": (time.time() - start) / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
    embeddings = download_hugging_face_embeddings()

    def store(namespace):
        return PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings,
                                                       namespace=namespace)

    new_store = store(NAMESPACE)
    setups = [
        ("Old: 300-char chunks, vector only", build_retriever(store(LEGACY_NAMESPACE), use_bm25=False, use_reranker=False)),
        ("1000-char chunks, vector only", build_retriever(new_store, use_bm25=False, use_reranker=False)),
        ("+ BM25 keyword search (hybrid)", build_retriever(new_store, use_bm25=True, use_reranker=False)),
        ("+ cross-encoder reranker (app)", build_retriever(new_store, use_bm25=True, use_reranker=True)),
    ]

    results = [evaluate(name, r, questions, FINAL_K, args.verbose) for name, r in setups]

    print(f"\nRetrieval quality on {len(questions)} questions (top {FINAL_K} passages):\n")
    print(f"{'Setup':<38} {'Hit@' + str(FINAL_K):>7} {'MRR':>6} {'sec/q':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['setup']:<38} {r['hit']:>7.0%} {r['mrr']:>6.2f} {r['secs']:>6.2f}")


if __name__ == "__main__":
    main()
