"""Hybrid retrieval: vector search + BM25 keyword search, fused, then reranked by a cross-encoder.

Why each stage:
- Vector search (Pinecone) finds passages with the same *meaning*, even with different wording.
- BM25 finds passages with the same *words*, which matters for drug names and medical terms
  (e.g. "metformin", "succinylcholine") that embeddings often blur together.
- The cross-encoder reads the question and each candidate passage *together* and scores how
  well the passage answers the question, which is far more accurate than either search alone.
"""
import logging
import os
import re

from typing import Sequence

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict
from langchain_community.retrievers import BM25Retriever

from src.config import (CANDIDATES_PER_RETRIEVER, CHUNKS_PATH, FINAL_K, MIN_RELEVANCE_SCORE,
                        RERANKER_MODEL, VECTOR_WEIGHT)
from src.helper import load_chunks

log = logging.getLogger(__name__)

_STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its
me my of on or should that the their them there these they this to was what when where which who
why will with you your""".split())


def bm25_tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def build_bm25_retriever(chunks, k=CANDIDATES_PER_RETRIEVER):
    retriever = BM25Retriever.from_documents(chunks, preprocess_func=bm25_tokenize)
    retriever.k = k
    return retriever


class ScoredCrossEncoderReranker(BaseDocumentCompressor):
    """Like LangChain's CrossEncoderReranker, but keeps each passage's score in
    metadata["relevance_score"] so the app can refuse to answer when even the best
    passage is a poor match for the question."""

    model: BaseCrossEncoder
    top_n: int = FINAL_K

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks=None):
        if not documents:
            return []
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)[: self.top_n]
        # Copy documents so the shared in-memory BM25 corpus is never mutated
        return [Document(page_content=doc.page_content, id=doc.id,
                         metadata={**doc.metadata, "relevance_score": float(score)})
                for doc, score in ranked]


def build_reranker(top_n=FINAL_K):
    return ScoredCrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL), top_n=top_n)


class HybridRetriever:
    """What the app uses: vector + BM25 search (optionally limited to some collections),
    merged with weighted Reciprocal Rank Fusion, then reranked by the cross-encoder.

    LangChain's EnsembleRetriever can't apply a per-request collection filter, so the
    fusion is done here directly. Each returned passage has metadata["relevance_score"].
    """

    def __init__(self, vector_store, chunks_path=CHUNKS_PATH, candidates=CANDIDATES_PER_RETRIEVER,
                 final_k=FINAL_K, vector_weight=VECTOR_WEIGHT):
        from rank_bm25 import BM25Okapi

        self.vector_store = vector_store
        self.candidates = candidates
        self.final_k = final_k
        self.vector_weight = vector_weight
        self.reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        self.chunks = load_chunks(chunks_path) if os.path.exists(chunks_path) else []
        self.bm25 = BM25Okapi([bm25_tokenize(c.page_content) for c in self.chunks]) if self.chunks else None
        if not self.chunks:
            log.warning("%s not found; keyword search disabled. Run store_index.py to enable it.", chunks_path)

    def _keyword_search(self, query, collections):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(bm25_tokenize(query))
        results = []
        for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=True):
            if scores[i] <= 0 or len(results) >= self.candidates:
                break
            chunk = self.chunks[i]
            if collections and chunk.metadata.get("collection") not in collections:
                continue
            results.append(chunk)
        return results

    def retrieve(self, query, collections=None, k=None):
        collections = [c for c in (collections or []) if c]
        search_filter = {"collection": {"$in": collections}} if collections else None
        vector_hits = self.vector_store.similarity_search(query, k=self.candidates, filter=search_filter)
        keyword_hits = self._keyword_search(query, collections)

        # Weighted Reciprocal Rank Fusion: passages ranked high by either search rise to the top
        fused = {}
        for hits, weight in ((vector_hits, self.vector_weight), (keyword_hits, 1 - self.vector_weight)):
            for rank, doc in enumerate(hits):
                key = doc.page_content
                entry = fused.setdefault(key, [doc, 0.0])
                entry[1] += weight / (60 + rank)
        candidates = [doc for doc, _ in sorted(fused.values(), key=lambda e: e[1], reverse=True)]
        if not candidates:
            return []

        scores = self.reranker.score([(query, d.page_content) for d in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[: k or self.final_k]
        return [Document(page_content=d.page_content, id=d.id,
                         metadata={**d.metadata, "relevance_score": float(s)}) for d, s in ranked]

    def score(self, query, texts):
        """Cross-encoder relevance of arbitrary texts (used to pick parts of uploaded documents)."""
        return [float(s) for s in self.reranker.score([(query, t) for t in texts])] if texts else []

    def invoke(self, query):
        return self.retrieve(query)

    @staticmethod
    def confident(docs):
        """Is the best passage a good enough match to answer from?"""
        best = docs[0].metadata.get("relevance_score") if docs else None
        return bool(docs) and (best is None or best >= MIN_RELEVANCE_SCORE)


def build_retriever(vector_store, chunks_path=CHUNKS_PATH, use_bm25=True, use_reranker=True,
                    final_k=FINAL_K):
    """Return the best retriever available.

    Falls back gracefully: if the local chunk file is missing (store_index.py not run yet),
    keyword search is skipped and only vector search + reranking are used.
    """
    candidates = CANDIDATES_PER_RETRIEVER if use_reranker else final_k
    vector = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": candidates})

    base = vector
    if use_bm25:
        if os.path.exists(chunks_path):
            bm25 = build_bm25_retriever(load_chunks(chunks_path), k=candidates)
            base = EnsembleRetriever(retrievers=[vector, bm25], weights=[VECTOR_WEIGHT, 1 - VECTOR_WEIGHT])
        else:
            log.warning("%s not found; keyword search disabled. Run store_index.py to enable it.",
                        chunks_path)

    if not use_reranker:
        return base

    return ContextualCompressionRetriever(base_compressor=build_reranker(final_k), base_retriever=base)
