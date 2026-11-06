from flask import Flask, Response, abort, jsonify, render_template, request, session, stream_with_context
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from datetime import timedelta
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.config import (CHAT_RATE_LIMIT, INDEX_NAME, MIN_RELEVANCE_SCORE, MISTRAL_MIN_INTERVAL,
                        NAMESPACE, PROJECT_ROOT)
from src.helper import download_hugging_face_embeddings
from src.retrieval import build_retriever
from src import safety, storage
import json
import os
import re
import secrets
import threading
import time
import uuid

# Serve /static files at the site root too, so /favicon-32x32.png etc. resolve
app = Flask(__name__, static_folder='static', static_url_path='')
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

missing = [k for k, v in {'PINECONE_API_KEY': PINECONE_API_KEY, 'MISTRAL_API_KEY': MISTRAL_API_KEY}.items() if not v]
if missing:
    raise SystemExit(f"Missing environment variables: {', '.join(missing)}. Add them to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY


def _load_secret_key():
    """Use FLASK_SECRET_KEY, or a random key saved in instance/ so that users keep
    access to their saved chats across server restarts."""
    if os.environ.get("FLASK_SECRET_KEY"):
        return os.environ["FLASK_SECRET_KEY"]
    path = os.path.join(PROJECT_ROOT, "instance", "secret_key")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.secret_key = _load_secret_key()
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=365)
storage.init_db()

# Per-visitor rate limit on chat endpoints (in memory; resets when the server restarts)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({
        "response": "⏳ You're sending messages too quickly. Please wait a minute and try again.",
        "kind": "rate_limited",
    }), 429


# ── RAG pipeline ─────────────────────────────────────────────────────────────
embeddings = download_hugging_face_embeddings()

system_prompt = (
    "You are Dr. Insight, a knowledgeable and empathetic AI medical assistant. "
    "Your role is to provide accurate, evidence-based medical information based strictly on the provided context. "
    "Be concise, clear, and compassionate. If the context does not contain enough information to answer, say so honestly. "
    "Never fabricate medical facts. Always recommend consulting a real doctor for diagnosis or treatment. "
    "Never give specific doses or dosing schedules, never tell the user which medicine they personally "
    "should take, start, stop or change, and never diagnose the user; you may explain what medicines are "
    "generally used for, how they work, and their side effects and interactions. "
    "Only when the user expresses worry, fear or other feelings, briefly acknowledge them first; "
    "for factual questions, answer directly without assuming how the user feels. "
    "Format answers in Markdown: use short paragraphs, bullet lists and **bold** for key terms.\n"
    "The reference passages below were retrieved from medical reference books by the system; the user did not "
    "write or provide them, so never refer to them as something the user said or shared. "
    "Each passage starts with its book and page in square brackets. Don't add citations to your answer; "
    "the sources are shown to the user separately. Ignore passages that are not relevant to the question. "
    "For questions about the conversation itself, rely only on the chat messages.\n\n"
    "Reference passages:\n{context}"
)

# Turns a follow-up like "what are its treatments?" into a standalone search query
# ("what are the treatments for diabetes?") so retrieval finds the right passages.
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation so far and the user's latest message, rewrite the latest message "
     "as a standalone question that can be understood without the conversation. "
     "Resolve pronouns and references (it, this, that condition, etc.). "
     "Do NOT answer the question. If it is already standalone, return it unchanged. "
     "Return only the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Namespace and index come from src/config.py, shared with store_index.py
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# Vector search + BM25 keyword search, fused and reranked (see src/retrieval.py)
retriever = build_retriever(docsearch)

llm = ChatMistralAI(
    # Override via MISTRAL_MODEL in .env (e.g. mistral-large-latest on a paid tier)
    model=os.environ.get("MISTRAL_MODEL", "open-mistral-nemo"),
    temperature=0.2,
    api_key=MISTRAL_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Label each passage with its book and page so the model knows where facts come from
document_prompt = PromptTemplate.from_template("[{source}, page {page}]\n{page_content}")
question_answer_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

MAX_HISTORY_MESSAGES = 8      # recent messages sent to the LLM as context
FALLBACK_ANSWER = ("I don't have enough information to answer that accurately. Could you rephrase, "
                   "or consult a healthcare professional for this concern?")
ERROR_ANSWER = "I'm having trouble processing that request. Please try again."


_mistral_lock = threading.Lock()
_last_mistral_call = 0.0


def wait_for_mistral_slot():
    """Space Mistral calls at least MISTRAL_MIN_INTERVAL apart across all users and threads.
    The free tier allows ~1 request/second, so this prevents most 429 errors up front."""
    global _last_mistral_call
    with _mistral_lock:
        wait = _last_mistral_call + MISTRAL_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_mistral_call = time.monotonic()


def invoke_with_retry(chain, inputs, attempts=3):
    """Throttle Mistral calls, and back off briefly if a 429 still happens."""
    for attempt in range(attempts):
        try:
            wait_for_mistral_slot()
            return chain.invoke(inputs)
        except Exception as e:
            if "429" in str(e) and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def stream_with_retry(chain, inputs, attempts=3):
    """Like invoke_with_retry, but yields text chunks as they arrive. A 429 can only be
    retried before the first chunk; after that, partial text has already been shown."""
    for attempt in range(attempts):
        started = False
        try:
            wait_for_mistral_slot()
            for chunk in chain.stream(inputs):
                if chunk:
                    started = True
                    yield chunk
            return
        except Exception as e:
            if "429" in str(e) and not started and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


# ── Quick replies for pure greetings / small talk ────────────────────────────
# Only used when the *whole* message is small talk, so "hey, what causes diabetes?"
# still gets a real answer.
QUICK_RESPONSES = {
    "hi": "Hello! I'm Dr. Insight, your AI medical assistant. How can I help you today?",
    "hello": "Hi there! What health questions can I help you with?",
    "hey": "Hey! I'm here to help with your health-related questions.",
    "how are you": "I'm an AI, so I don't have feelings — but I'm fully operational and ready to assist you!",
    "thank you": "You're welcome! Feel free to ask anything else.",
    "thanks": "Happy to help! Let me know if you need anything else.",
    "bye": "Take care of yourself! Don't hesitate to come back if you have more questions.",
    "goodbye": "Goodbye! Stay healthy!",
}

GREETING_ALIASES = {
    "hii": "hi", "hiii": "hi", "hi there": "hi", "hi doctor": "hi", "hi dr insight": "hi",
    "hello there": "hello", "hello doctor": "hello", "hello dr insight": "hello",
    "good morning": "hello", "good afternoon": "hello", "good evening": "hello",
    "hey there": "hey", "hey doctor": "hey",
    "how are you doing": "how are you", "how r u": "how are you", "hi how are you": "how are you",
    "hello how are you": "how are you", "hey how are you": "how are you",
    "thank you so much": "thank you", "thank you very much": "thank you",
    "thanks a lot": "thanks", "thanks so much": "thanks", "thx": "thanks", "ok thanks": "thanks",
    "okay thanks": "thanks", "ok thank you": "thank you",
    "bye bye": "bye", "see you": "bye", "see you later": "bye", "good bye": "goodbye",
}


def quick_reply(message):
    """Return a canned reply if the entire message is a greeting/thanks/farewell."""
    normalized = re.sub(r"[^a-z\s]", " ", message.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    key = GREETING_ALIASES.get(normalized, normalized)
    return QUICK_RESPONSES.get(key)


def format_sources(docs):
    """Unique (book, page) pairs for the passages the answer was based on, best match first."""
    sources, seen = [], set()
    for doc in docs:
        title = doc.metadata.get("source", "Unknown source")
        page = str(doc.metadata.get("page", "")).strip()
        if (title, page) in seen:
            continue
        seen.add((title, page))
        sources.append({"title": title, "page": page})
    return sources


def to_langchain_history(messages):
    """Saved messages -> LangChain messages. Failed turns are left out of the context."""
    history = []
    for m in messages:
        if m.get("kind") == "error":
            continue
        history.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    return history


# ── Answering a message (shared by the streaming and JSON endpoints) ─────────
def answer_events(conversation_id, user_message):
    """Yield (event, data) pairs for one user message and save both sides to the database.

    Events: "token" {"text"} for each piece of the answer, then "done" with the saved
    assistant message, or "error".
    """
    history = to_langchain_history(storage.get_messages(conversation_id, limit=MAX_HISTORY_MESSAGES))
    storage.add_message(conversation_id, "user", user_message)

    def finish(text, kind=None, sources=None, notice=None):
        message_id = storage.add_message(conversation_id, "assistant", text, sources=sources,
                                         kind=kind, notice=notice)
        return ("done", {"message_id": message_id, "content": text, "kind": kind,
                         "sources": sources or [], "notice": notice})

    # 1. Safety checks run before anything else, and skip the AI entirely when they trigger
    check = safety.check_message(user_message)
    canned = {
        safety.CRISIS: safety.crisis_response,
        safety.EMERGENCY: lambda: safety.emergency_response(check.category),
        safety.DOSAGE: safety.dosage_response,
    }.get(check.kind)
    if canned:
        app.logger.info("Safety rule '%s' (%s) triggered", check.kind, check.category or "-")
        text = canned()
        yield "token", {"text": text}
        yield finish(text, kind=check.kind)
        return

    # 2. Greetings and thanks
    reply = quick_reply(user_message)
    if reply:
        yield "token", {"text": reply}
        yield finish(reply)
        return

    parts, docs = [], []
    try:
        # 3. Rewrite follow-ups into standalone questions before searching. If the
        #    rewrite fails (e.g. rate limit), fall back to searching the raw message.
        search_query = user_message
        if history:
            try:
                rewritten = invoke_with_retry(contextualize_chain, {
                    "input": user_message, "chat_history": history}).strip()
                if rewritten:
                    search_query = rewritten
            except Exception as e:
                app.logger.warning("Question rewrite failed, using original message: %s", e)

        docs = retriever.invoke(search_query)

        # 4. Don't let the model guess when the reference books have nothing relevant
        best_score = docs[0].metadata.get("relevance_score") if docs else None
        if not docs or (best_score is not None and best_score < MIN_RELEVANCE_SCORE):
            app.logger.info("Low relevance (best score %s) for %r; declining to answer",
                            best_score, search_query)
            text = safety.low_confidence_response()
            yield "token", {"text": text}
            yield finish(text, kind="low_confidence", notice=check.notice)
            return

        # 5. Stream the answer from the LLM
        for chunk in stream_with_retry(question_answer_chain, {
                "input": user_message, "chat_history": history, "context": docs}):
            parts.append(chunk)
            yield "token", {"text": chunk}

        text = "".join(parts).strip()
        if not text:
            text = FALLBACK_ANSWER
            yield "token", {"text": text}
        yield finish(text, sources=format_sources(docs), notice=check.notice)

    except GeneratorExit:
        # The browser stopped the request (e.g. the "Stop" button); keep what was written
        partial = "".join(parts).strip()
        if partial:
            storage.add_message(conversation_id, "assistant", partial + "\n\n*(stopped)*",
                                sources=format_sources(docs))
        raise
    except Exception as e:
        app.logger.exception("Error generating response: %s", e)
        yield "error", {"text": ERROR_ANSWER}
        storage.add_message(conversation_id, "assistant", ERROR_ANSWER, kind="error")


# ── Request helpers ──────────────────────────────────────────────────────────
def current_user_id():
    """Anonymous per-browser id kept in the signed session cookie."""
    if "uid" not in session:
        session["uid"] = uuid.uuid4().hex
        session.permanent = True
    return session["uid"]


def _request_json():
    return request.get_json(silent=True) or {}


def _request_message():
    return str(_request_json().get("message", "")).strip()[:4000]


def _is_urgent_request():
    """Emergency and crisis messages are never rate-limited."""
    return safety.check_message(_request_message()).kind in (safety.EMERGENCY, safety.CRISIS)


def _conversation_for_request(user_id):
    """The conversation named in the request, or a new one if it's missing or not this user's."""
    conv_id = _request_json().get("conversation_id")
    conv = storage.get_conversation(user_id, conv_id) if conv_id else None
    return conv or storage.create_conversation(user_id)


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    current_user_id()
    return render_template('chat.html')


# ── Chat endpoints ───────────────────────────────────────────────────────────
@app.route("/api/chat/stream", methods=["POST"])
@limiter.limit(CHAT_RATE_LIMIT, exempt_when=_is_urgent_request)
def chat_stream():
    """Server-Sent Events: "conversation", then "token"s as the answer is written, then "done"."""
    user_id = current_user_id()
    message = _request_message()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    conv = _conversation_for_request(user_id)

    def generate():
        yield _sse("conversation", {"id": conv["id"], "title": storage.make_title(message)
                                    if conv["title"] == "New chat" else conv["title"]})
        for event, data in answer_events(conv["id"], message):
            yield _sse(event, data)

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/chat", methods=["POST"])
@limiter.limit(CHAT_RATE_LIMIT, exempt_when=_is_urgent_request)
def chat():
    """Non-streaming JSON API: {"message", "conversation_id"?} -> full answer at once."""
    user_id = current_user_id()
    message = _request_message()
    if not message:
        return jsonify({"response": "Please enter a message."})
    conv = _conversation_for_request(user_id)

    result = {"conversation_id": conv["id"]}
    for event, data in answer_events(conv["id"], message):
        if event == "done":
            result.update({"response": data["content"], "message_id": data["message_id"],
                           "sources": data["sources"], "notice": data["notice"], "kind": data["kind"]})
        elif event == "error":
            result.update({"response": data["text"], "kind": "error"})
    return jsonify(result)


# ── Conversation history & feedback ──────────────────────────────────────────
@app.route("/api/conversations")
def list_conversations():
    return jsonify(storage.list_conversations(current_user_id()))


@app.route("/api/conversations/<conv_id>")
def get_conversation(conv_id):
    conv = storage.get_conversation(current_user_id(), conv_id)
    if not conv:
        abort(404)
    return jsonify({**conv, "messages": storage.get_messages(conv_id)})


@app.route("/api/conversations/<conv_id>", methods=["PATCH"])
def rename_conversation(conv_id):
    title = str(_request_json().get("title", "")).strip()
    if not title:
        return jsonify({"error": "Title can't be empty."}), 400
    if not storage.rename_conversation(current_user_id(), conv_id, title):
        abort(404)
    return jsonify({"ok": True})


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    if not storage.delete_conversation(current_user_id(), conv_id):
        abort(404)
    return jsonify({"ok": True})


@app.route("/api/messages/<int:message_id>/feedback", methods=["POST"])
def message_feedback(message_id):
    value = _request_json().get("value")
    if value not in (1, -1, None):
        return jsonify({"error": "value must be 1, -1 or null"}), 400
    if not storage.set_feedback(current_user_id(), message_id, value):
        abort(404)
    if value == -1:
        app.logger.info("👎 feedback on message %s", message_id)
    return jsonify({"ok": True})


@app.route("/api/feedback/summary")
def feedback_summary():
    """👍/👎 totals and recent 👎 answers across all users. Only available from this computer."""
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403)
    return jsonify(storage.feedback_summary())


if __name__ == '__main__':
    import argparse
    from src.launcher import BROWSER_CHOICES, open_browser_when_ready

    parser = argparse.ArgumentParser(description="Run the Dr. Insight medical chatbot.")
    parser.add_argument("--browser", choices=BROWSER_CHOICES,
                        default=os.environ.get("OPEN_BROWSER", "default").lower(),
                        help="Browser to open on startup (default: your system's default browser; "
                             "'none' to not open one). Can also be set with OPEN_BROWSER in .env.")
    args = parser.parse_args()
    if args.browser not in BROWSER_CHOICES:
        parser.error(f"OPEN_BROWSER must be one of: {', '.join(BROWSER_CHOICES)}")

    # Debug mode is opt-in: Werkzeug's debugger lets anyone who can reach it run code
    # on this machine, so it must never be combined with a network-visible host.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    host = os.environ.get("HOST", "127.0.0.1")
    if debug and host not in ("127.0.0.1", "localhost"):
        print(f"FLASK_DEBUG is on, so binding to 127.0.0.1 instead of {host} for safety.")
        host = "127.0.0.1"
    port = int(os.environ.get("PORT", 8080))

    # With debug's auto-reloader the script runs twice; only open from the parent process.
    if args.browser != "none" and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        url = f"http://localhost:{port}"
        print(f"Opening {url} in {'your default browser' if args.browser == 'default' else args.browser.title()}...")
        open_browser_when_ready(url, args.browser)

    app.run(host=host, port=port, debug=debug, threaded=True)
