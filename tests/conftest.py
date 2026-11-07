"""Shared test fixtures: a Flask test client wired to fake AI services.

No API keys, network or model downloads are needed: the LLM, retriever and vision model
are replaced by small fakes that return scripted answers and record how they were called.
"""
import pytest
from langchain_core.documents import Document

import app as app_module
from src import storage


class FakeChain:
    """Stands in for a LangChain chain/LLM: .invoke() returns text, .stream() yields words."""

    def __init__(self, reply="**Diabetes** is a condition with high blood sugar.", errors=None):
        self.reply = reply
        self.errors = list(errors or [])   # exceptions to raise on the next calls, in order
        self.calls = []

    def _maybe_fail(self):
        if self.errors:
            raise self.errors.pop(0)

    def invoke(self, inputs):
        self.calls.append(inputs)
        self._maybe_fail()
        return self.reply

    def stream(self, inputs):
        self.calls.append(inputs)
        self._maybe_fail()
        for word in self.reply.split(" "):
            yield word + " "


class FakeVision(FakeChain):
    def invoke(self, inputs):
        text = super().invoke(inputs)
        return type("Message", (), {"content": text})()


class FakeRetriever:
    def __init__(self, score=5.0):
        self.score_value = score
        self.queries = []

    def retrieve(self, query, collections=None, k=None):
        self.queries.append((query, collections))
        return [Document(page_content="Diabetes means blood sugar is too high.",
                         metadata={"source": "MedlinePlus: Diabetes", "page": "", "citation": "MedlinePlus: Diabetes",
                                   "url": "https://medlineplus.gov/diabetes.html", "relevance_score": self.score_value})]

    def score(self, query, texts):
        return [1.0] * len(texts)

    @staticmethod
    def confident(docs):
        from src.retrieval import HybridRetriever
        return HybridRetriever.confident(docs)


class FakeServices:
    def __init__(self):
        self.retriever = FakeRetriever()
        self.qa_chain = FakeChain()
        self.symptom_chain = FakeChain("### Possible causes\n- **Common cold**")
        self.contextualize_chain = FakeChain("TRANSLATION: What are the symptoms of diabetes?\n"
                                             "QUESTION: What are the symptoms of diabetes?")
        self.translate_chain = FakeChain("अनुवादित उत्तर")
        self.vision_llm = FakeVision("| Test | Value |\n|---|---|\n| Haemoglobin | 11.2 |")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Fresh database, fake services, no rate limiting or Mistral pacing delays."""
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test.db"))
    storage.init_db()
    services = FakeServices()
    monkeypatch.setattr(app_module, "_services", services)
    monkeypatch.setattr(app_module, "MISTRAL_MIN_INTERVAL", 0)
    monkeypatch.setattr(app_module.time, "sleep", lambda seconds: None)
    app_module.limiter.enabled = False
    yield services
    app_module.limiter.enabled = True
    app_module.limiter.reset()


@pytest.fixture
def client(fake):
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def ask(client, message, **extra):
    response = client.post("/chat", json={"message": message, **extra})
    return response.status_code, response.get_json()
