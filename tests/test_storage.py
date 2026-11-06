"""Tests for saved conversations and feedback (uses a temporary database)."""
import pytest

from src import storage


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test.db"))
    storage.init_db()


def test_conversation_title_comes_from_first_message():
    conv = storage.create_conversation("alice")
    storage.add_message(conv["id"], "user", "What are the side effects of metformin?")
    storage.add_message(conv["id"], "user", "And what about insulin?")
    assert storage.get_conversation("alice", conv["id"])["title"] == "What are the side effects of metformin?"


def test_long_titles_are_shortened():
    assert len(storage.make_title("word " * 50)) <= 60


def test_users_only_see_their_own_conversations():
    a = storage.create_conversation("alice")
    storage.create_conversation("bob")
    assert [c["id"] for c in storage.list_conversations("alice")] == [a["id"]]
    assert storage.get_conversation("bob", a["id"]) is None
    assert storage.delete_conversation("bob", a["id"]) is False


def test_messages_keep_sources_and_order():
    conv = storage.create_conversation("alice")
    storage.add_message(conv["id"], "user", "q1")
    storage.add_message(conv["id"], "assistant", "a1", sources=[{"title": "Book", "page": "5"}])
    storage.add_message(conv["id"], "user", "q2")
    messages = storage.get_messages(conv["id"])
    assert [m["content"] for m in messages] == ["q1", "a1", "q2"]
    assert messages[1]["sources"] == [{"title": "Book", "page": "5"}]
    assert [m["content"] for m in storage.get_messages(conv["id"], limit=2)] == ["a1", "q2"]


def test_feedback_only_on_own_assistant_messages():
    conv = storage.create_conversation("alice")
    question = storage.add_message(conv["id"], "user", "q")
    answer = storage.add_message(conv["id"], "assistant", "a")
    assert storage.set_feedback("alice", answer, -1)
    assert not storage.set_feedback("alice", question, 1)      # not an answer
    assert not storage.set_feedback("bob", answer, 1)          # not bob's
    summary = storage.feedback_summary()
    assert summary["down"] == 1 and summary["recent_down"][0]["question"] == "q"


def test_deleting_a_conversation_deletes_its_messages():
    conv = storage.create_conversation("alice")
    storage.add_message(conv["id"], "user", "q")
    assert storage.delete_conversation("alice", conv["id"])
    assert storage.get_messages(conv["id"]) == []
