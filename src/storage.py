"""Saved conversations, messages and 👍/👎 feedback in a small SQLite database.

The database file lives in instance/medirag.db (git-ignored). Each browser gets an
anonymous user id in its session cookie, and only sees its own conversations.
"""
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from src.config import PROJECT_ROOT

DB_PATH = os.environ.get("CHAT_DB_PATH", os.path.join(PROJECT_ROOT, "instance", "medirag.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    sources          TEXT,               -- JSON list of {title, page, url}
    kind             TEXT,               -- emergency / crisis / dosage / low_confidence / ...
    notice           TEXT,
    feedback         INTEGER,            -- 1 = 👍, -1 = 👎, NULL = none
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS attachments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    filename         TEXT NOT NULL,
    kind             TEXT NOT NULL CHECK (kind IN ('pdf', 'image')),
    path             TEXT NOT NULL,      -- file on disk under instance/uploads/
    text             TEXT NOT NULL,      -- extracted text, or the AI's description of an image
    pages            INTEGER,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments(conversation_id);
"""



def upload_dir():
    """Folder for uploaded files, next to the database (instance/uploads/)."""
    return os.path.join(os.path.dirname(DB_PATH), "uploads")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(upload_dir(), exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


def make_title(first_message, limit=60):
    title = " ".join(first_message.split())
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


# ── Conversations ────────────────────────────────────────────────────────────
def create_conversation(user_id, title="New chat"):
    now = time.time()
    conv_id = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute("INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
                     "VALUES (?, ?, ?, ?, ?)", (conv_id, user_id, title, now, now))
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def get_conversation(user_id, conv_id):
    with _connect() as conn:
        row = conn.execute("SELECT id, title, created_at, updated_at FROM conversations "
                           "WHERE id = ? AND user_id = ?", (conv_id, user_id)).fetchone()
    return dict(row) if row else None


def list_conversations(user_id, limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, "
            "       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
            "FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC LIMIT ?",
            (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(user_id, conv_id, title):
    with _connect() as conn:
        cur = conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
                           (make_title(title, 80), conv_id, user_id))
    return cur.rowcount > 0


def delete_conversation(user_id, conv_id):
    with _connect() as conn:
        paths = [r["path"] for r in conn.execute(
            "SELECT a.path FROM attachments a JOIN conversations c ON c.id = a.conversation_id "
            "WHERE c.id = ? AND c.user_id = ?", (conv_id, user_id))]
        cur = conn.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
    # Uploaded medical documents are sensitive: remove the files too, not just the rows
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
    return cur.rowcount > 0


# ── Attachments (uploaded PDFs and images) ───────────────────────────────────
def add_attachment(conv_id, filename, kind, path, text, pages=None):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO attachments (conversation_id, filename, kind, path, text, pages, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (conv_id, filename, kind, path, text, pages, time.time()))
    return cur.lastrowid


def list_attachments(conv_id):
    with _connect() as conn:
        rows = conn.execute("SELECT id, filename, kind, path, text, pages, created_at FROM attachments "
                            "WHERE conversation_id = ? ORDER BY id", (conv_id,)).fetchall()
    return [dict(r) for r in rows]


def get_attachment(user_id, attachment_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT a.* FROM attachments a JOIN conversations c ON c.id = a.conversation_id "
            "WHERE a.id = ? AND c.user_id = ?", (attachment_id, user_id)).fetchone()
    return dict(row) if row else None


# ── Messages ─────────────────────────────────────────────────────────────────
def add_message(conv_id, role, content, sources=None, kind=None, notice=None):
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, kind, notice, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conv_id, role, content, json.dumps(sources) if sources else None, kind, notice, now))
        # The first user message becomes the conversation title
        if role == "user":
            conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = 'New chat'",
                         (make_title(content), conv_id))
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
    return cur.lastrowid


def _message_dict(row):
    d = dict(row)
    d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
    return d


def get_messages(conv_id, limit=None):
    """Messages of a conversation in order. With limit, only the most recent ones."""
    with _connect() as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM (SELECT id, role, content, sources, kind, notice, feedback, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (conv_id, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, role, content, sources, kind, notice, feedback, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY id", (conv_id,)).fetchall()
    return [_message_dict(r) for r in rows]


def set_feedback(user_id, message_id, value):
    """value: 1 (👍), -1 (👎) or None (clear). Only assistant messages the user owns."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE messages SET feedback = ? WHERE id = ? AND role = 'assistant' AND conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id = ?)", (value, message_id, user_id))
    return cur.rowcount > 0


def feedback_summary():
    """Counts of 👍 and 👎 across all users, plus the most recent 👎 answers for review."""
    with _connect() as conn:
        counts = {r["feedback"]: r["n"] for r in conn.execute(
            "SELECT feedback, COUNT(*) AS n FROM messages WHERE feedback IS NOT NULL GROUP BY feedback")}
        bad = conn.execute(
            "SELECT a.id, a.content AS answer, a.created_at, "
            "  (SELECT q.content FROM messages q WHERE q.conversation_id = a.conversation_id "
            "     AND q.id < a.id AND q.role = 'user' ORDER BY q.id DESC LIMIT 1) AS question "
            "FROM messages a WHERE a.feedback = -1 ORDER BY a.id DESC LIMIT 20").fetchall()
    return {"up": counts.get(1, 0), "down": counts.get(-1, 0), "recent_down": [dict(r) for r in bad]}
