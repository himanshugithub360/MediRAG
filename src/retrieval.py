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

from src.config import (CANDIDATES_PER_RETRIEVER, CHUNKS_PATH, FINAL_K, RERANKER_MODEL,
                        VECTOR_WEIGHT)
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
