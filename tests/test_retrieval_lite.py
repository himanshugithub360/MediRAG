"""Tests for the lightweight (Render) retriever, with fake Pinecone and embedding objects."""
from types import SimpleNamespace as NS

import pytest

from src import config
from src.retrieval_lite import LiteRetriever


class FakeEmbedder:
    def embed(self, texts):
        for _ in texts:
            yield [0.1] * 384


class FakeIndex:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        texts = ["Metformin side effects include nausea.", "Asthma is a lung disease.", "Diabetes overview."]
        return NS(matches=[NS(id=f"id{i}", score=0.7 - i * 0.1,
                              metadata={"text": t, "source": f"Book {i}", "collection": "pharmacology"})
                           for i, t in enumerate(texts)])


class FakeReranker:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def rerank(self, model, query, documents, top_n, return_documents):
        self.calls += 1
        if self.error:
            raise self.error
        # Reverse the order to show the reranker's order wins
        return NS(data=[NS(index=i, score=0.9 - n * 0.1) for n, i in enumerate(reversed(range(len(documents))))][:top_n])


def make(reranker=None):
    index = FakeIndex()
    return LiteRetriever(index=index, embedder=FakeEmbedder(), reranker=reranker), index


def test_hosted_rerank_order_and_scores(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MODEL", "bge-reranker-v2-m3")
    r, _ = make(FakeReranker())
    docs = r.retrieve("diabetes", k=2)
    assert [d.page_content for d in docs] == ["Diabetes overview.", "Asthma is a lung disease."]
    assert docs[0].metadata["score_type"] == "rerank" and docs[0].metadata["relevance_score"] == 0.9
    assert "text" not in docs[0].metadata


def test_falls_back_to_vector_order_when_reranker_quota_is_used_up(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MODEL", "bge-reranker-v2-m3")
    reranker = FakeReranker(error=RuntimeError("(429) monthly rerank quota exceeded"))
    r, _ = make(reranker)
    docs = r.retrieve("metformin", k=2)
    assert docs[0].page_content.startswith("Metformin")
    assert docs[0].metadata["score_type"] == "vector" and docs[0].metadata["relevance_score"] == pytest.approx(0.7)
    r.retrieve("metformin", k=2)
    assert reranker.calls == 1          # paused after the quota error instead of retrying every time


def test_collection_filter_is_sent_to_pinecone(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MODEL", "")
    r, index = make()
    r.retrieve("malaria", collections=["who"])
    assert index.calls[-1]["filter"] == {"collection": {"$in": ["who"]}}
    r.retrieve("malaria")
    assert index.calls[-1]["filter"] is None


def test_confidence_thresholds_depend_on_score_type():
    from langchain_core.documents import Document

    def doc(score, kind):
        return [Document(page_content="x", metadata={"relevance_score": score, "score_type": kind})]

    assert LiteRetriever.confident(doc(0.9, "rerank"))
    assert not LiteRetriever.confident(doc(0.03, "rerank"))     # off-topic questions score ~0.0-0.04
    assert LiteRetriever.confident(doc(0.6, "vector"))
    assert not LiteRetriever.confident(doc(0.3, "vector"))      # off-topic questions score ~0.2-0.32
    assert not LiteRetriever.confident([])


def test_keyword_score_for_uploads():
    scores = LiteRetriever.score("haemoglobin value", ["Haemoglobin 11.2 g/dL low", "Cholesterol 185"])
    assert scores[0] > scores[1]
