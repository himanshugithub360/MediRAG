"""Lightweight retrieval for small servers (Render free tier: 512 MB RAM, 0.1 CPU).

No PyTorch and no local reranker:
- The question is embedded with the same all-MiniLM-L6-v2 model as the index, but run
  through ONNX Runtime (fastembed): identical vectors, ~10x less memory than PyTorch.
- Pinecone vector search returns the top candidates (optionally limited to collections).
- Pinecone's *hosted* reranker (free plan: 500 requests/month) picks the best 5. When it's
  unavailable or the monthly allowance is used up, the vector-search order is used instead.

Relevance scores differ by stage, so each passage carries metadata["score_type"]:
"rerank" (0..1 from the hosted reranker) or "vector" (cosine similarity).
"""
import logging
import re
import threading
import time

from langchain_core.documents import Document

from src import config

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class LiteRetriever:
    def __init__(self, index=None, embedder=None, reranker=None):
        if index is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            index = pc.Index(config.INDEX_NAME)
            reranker = reranker if reranker is not None else pc.inference
        if embedder is None:
            from fastembed import TextEmbedding

            embedder = TextEmbedding(config.EMBEDDING_MODEL, cache_dir=config.FASTEMBED_CACHE or None)
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self._rerank_paused_until = 0.0
        self._lock = threading.Lock()

    # ── Search ───────────────────────────────────────────────────────────────
    def _embed(self, text):
        return [float(x) for x in next(iter(self.embedder.embed([text])))]

    def _vector_search(self, query, collections):
        result = self.index.query(
            vector=self._embed(query), top_k=config.CANDIDATES_PER_RETRIEVER, namespace=config.NAMESPACE,
            filter={"collection": {"$in": collections}} if collections else None, include_metadata=True)
        docs = []
        for match in result.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop("text", "")
            if text:
                docs.append(Document(page_content=text, id=match.id,
                                     metadata={**metadata, "vector_score": float(match.score)}))
        return docs

    def _rerank(self, query, docs, k):
        """Hosted reranking, or None if it's unavailable (e.g. the free monthly allowance is used up)."""
        if not config.RERANK_MODEL or self.reranker is None or time.time() < self._rerank_paused_until:
            return None
        try:
            result = self.reranker.rerank(model=config.RERANK_MODEL, query=query,
                                          documents=[d.page_content for d in docs], top_n=k,
                                          return_documents=False)
            return [(docs[item.index], float(item.score)) for item in result.data]
        except Exception as e:
            # Don't retry on every question: wait an hour (quota) or a minute (other errors)
            quota = "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower()
            with self._lock:
                self._rerank_paused_until = time.time() + (3600 if quota else 60)
            log.warning("Hosted reranking unavailable (%s); using vector order. %s",
                        "allowance used up" if quota else "error", str(e)[:200])
            return None

    def retrieve(self, query, collections=None, k=None):
        k = k or config.FINAL_K
        collections = [c for c in (collections or []) if c]
        docs = self._vector_search(query, collections)
        if not docs:
            return []
        ranked = self._rerank(query, docs, k)
        if ranked is not None:
            return [Document(page_content=d.page_content, id=d.id,
                             metadata={**d.metadata, "relevance_score": s, "score_type": "rerank"})
                    for d, s in ranked]
        return [Document(page_content=d.page_content, id=d.id,
                         metadata={**d.metadata, "relevance_score": d.metadata["vector_score"],
                                   "score_type": "vector"})
                for d in docs[:k]]

    def invoke(self, query):
        return self.retrieve(query)

    @staticmethod
    def confident(docs):
        """Is the best passage a good enough match to answer from?"""
        if not docs:
            return False
        md = docs[0].metadata
        threshold = config.MIN_RERANK_SCORE if md.get("score_type") == "rerank" else config.MIN_VECTOR_SCORE
        return md.get("relevance_score", 0.0) >= threshold

    @staticmethod
    def score(query, texts):
        """Cheap keyword-overlap relevance for picking parts of long uploaded documents
        (running a neural model over them would be far too slow on 0.1 CPU)."""
        terms = set(_TOKEN.findall(query.lower())) - {"the", "a", "an", "of", "is", "are", "my", "what", "and"}
        scores = []
        for text in texts:
            words = _TOKEN.findall(text.lower())
            scores.append(sum(1 for w in words if w in terms) / (1 + len(words)) ** 0.5)
        return scores
