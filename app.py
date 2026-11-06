from flask import Flask, jsonify, request, render_template, session
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from collections import OrderedDict
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.config import (CHAT_RATE_LIMIT, INDEX_NAME, MIN_RELEVANCE_SCORE, MISTRAL_MIN_INTERVAL,
                        NAMESPACE)
from src.helper import download_hugging_face_embeddings
from src.retrieval import build_retriever
from src import safety
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
# Without a fixed key, sessions reset whenever the server restarts — fine for local use
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)

# Per-visitor rate limit on /chat (in memory; resets when the server restarts)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({
        "response": "⏳ You're sending messages too quickly. Please wait a minute and try again.",
        "kind": "rate_limited",
    }), 429


missing = [k for k, v in {'PINECONE_API_KEY': PINECONE_API_KEY, 'MISTRAL_API_KEY': MISTRAL_API_KEY}.items() if not v]
if missing:
    raise SystemExit(f"Missing environment variables: {', '.join(missing)}. Add them to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

embeddings = download_hugging_face_embeddings()

system_prompt = (
    "You are Dr. Insight, a knowledgeable and empathetic AI medical assistant. "
    "Your role is to provide accurate, evidence-based medical information based strictly on the provided context. "
    "Be concise, clear, and compassionate. If the context does not contain enough information to answer, say so honestly. "
    "Never fabricate medical facts. Always recommend consulting a real doctor for diagnosis or treatment. "
    "Never give specific doses or dosing schedules, never tell the user which medicine they personally "
    "should take, start, stop or change, and never diagnose the user; you may explain what medicines are "
    "generally used for, how they work, and their side effects and interactions. "
    "For emotional statements, acknowledge the person's feelings first before providing information.\n"
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


# ── Conversation memory (server-side) ────────────────────────────────────────
# History used to live in Flask's cookie session, which is capped at ~4 KB; a few
# long answers overflowed it and the conversation was silently lost. Now the cookie
# only carries a conversation id and the messages live in server memory.
MAX_HISTORY_MESSAGES = 8      # messages sent to the LLM as context
MAX_STORED_MESSAGES = 20      # messages kept per conversation
MAX_CONVERSATIONS = 1000      # oldest conversations are evicted beyond this

_conversations: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
_conversations_lock = threading.Lock()


def get_history(conv_id):
    with _conversations_lock:
        if conv_id in _conversations:
            _conversations.move_to_end(conv_id)
            return list(_conversations[conv_id])
        return []


def append_history(conv_id, user_message, bot_response):
    with _conversations_lock:
        history = _conversations.setdefault(conv_id, [])
        history.append(("human", user_message))
        history.append(("ai", bot_response))
        del history[:-MAX_STORED_MESSAGES]
        _conversations.move_to_end(conv_id)
        while len(_conversations) > MAX_CONVERSATIONS:
            _conversations.popitem(last=False)


def to_messages(history):
    return [HumanMessage(text) if role == "human" else AIMessage(text) for role, text in history]


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


# ── Quick replies for pure greetings / small talk ────────────────────────────
# Only used when the *whole* message is small talk. Previously "hey, what causes
# diabetes?" matched "hey" and the medical question was never answered.
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


def get_conversation_id():
    if "conv_id" not in session:
        session["conv_id"] = uuid.uuid4().hex
    return session["conv_id"]


@app.route("/")
def index():
    return render_template('chat.html')


def _request_message():
    data = request.get_json(silent=True) or {}
    return str(data.get("message", "")).strip()


def _is_urgent_request():
    """Emergency and crisis messages are never rate-limited."""
    return safety.check_message(_request_message()).kind in (safety.EMERGENCY, safety.CRISIS)


@app.route("/chat", methods=["POST"])
@limiter.limit(CHAT_RATE_LIMIT, exempt_when=_is_urgent_request)
def chat():
    user_message = _request_message()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    conv_id = get_conversation_id()

    # Safety checks run before anything else, and skip the AI entirely when they trigger
    check = safety.check_message(user_message)
    canned = {
        safety.CRISIS: safety.crisis_response,
        safety.EMERGENCY: lambda: safety.emergency_response(check.category),
        safety.DOSAGE: safety.dosage_response,
    }.get(check.kind)
    if canned:
        reply = canned()
        app.logger.info("Safety rule '%s' (%s) triggered", check.kind, check.category or "-")
        append_history(conv_id, user_message, reply)
        return jsonify({"response": reply, "kind": check.kind})

    reply = quick_reply(user_message)
    if reply:
        append_history(conv_id, user_message, reply)
        return jsonify({"response": reply})

    chat_history = to_messages(get_history(conv_id)[-MAX_HISTORY_MESSAGES:])

    try:
        # Rewrite follow-ups into standalone questions before searching. If the
        # rewrite fails (e.g. rate limit), fall back to searching the raw message.
        search_query = user_message
        if chat_history:
            try:
                rewritten = invoke_with_retry(contextualize_chain, {
                    "input": user_message,
                    "chat_history": chat_history,
                }).strip()
                if rewritten:
                    search_query = rewritten
            except Exception as e:
                app.logger.warning("Question rewrite failed, using original message: %s", e)

        docs = retriever.invoke(search_query)

        # Don't let the model guess when the reference books have nothing relevant
        best_score = docs[0].metadata.get("relevance_score") if docs else None
        if not docs or (best_score is not None and best_score < MIN_RELEVANCE_SCORE):
            app.logger.info("Low relevance (best score %s) for %r; declining to answer",
                            best_score, search_query)
            reply = safety.low_confidence_response()
            append_history(conv_id, user_message, reply)
            return jsonify({"response": reply, "kind": "low_confidence", "notice": check.notice})

        bot_response = invoke_with_retry(question_answer_chain, {
            "input": user_message,
            "chat_history": chat_history,
            "context": docs,
        }).strip()

        if not bot_response:
            bot_response = "I don't have enough information to answer that accurately. Could you rephrase, or consult a healthcare professional for this concern?"

    except Exception as e:
        app.logger.exception("Error generating response: %s", e)
        # Don't store failed turns, so they don't pollute the conversation context
        return jsonify({"response": "I'm having trouble processing that request. Please try again."})

    append_history(conv_id, user_message, bot_response)
    return jsonify({"response": bot_response, "sources": format_sources(docs), "notice": check.notice})


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


# ── Auto-open the chat in a browser on startup ───────────────────────────────
BROWSER_CHOICES = ("default", "chrome", "edge", "none")


def _find_browser(name):
    """Return a command (list) that launches Chrome or Edge, or None if not installed."""
    import shutil
    import sys

    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        rel = {"chrome": r"Google\Chrome\Application\chrome.exe",
               "edge": r"Microsoft\Edge\Application\msedge.exe"}[name]
        for base in (pf, pf86, local):
            path = os.path.join(base, rel)
            if base and os.path.isfile(path):
                return [path]
    elif sys.platform == "darwin":
        app_name = {"chrome": "Google Chrome", "edge": "Microsoft Edge"}[name]
        if os.path.isdir(f"/Applications/{app_name}.app"):
            return ["open", "-a", app_name]
    else:
        candidates = {"chrome": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"],
                      "edge": ["microsoft-edge", "microsoft-edge-stable"]}[name]
        for exe in candidates:
            if shutil.which(exe):
                return [exe]
    return None


def open_browser(url, choice):
    import subprocess
    import webbrowser

    if choice in ("chrome", "edge"):
        cmd = _find_browser(choice)
        if cmd:
            subprocess.Popen(cmd + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        print(f"{choice.title()} was not found; opening your default browser instead.")
    webbrowser.open(url)


def open_browser_when_ready(url, choice, timeout=120):
    """Wait until the server answers, then open the browser (runs in a background thread)."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    open_browser(url, choice)


if __name__ == '__main__':
    import argparse

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
        threading.Thread(target=open_browser_when_ready, args=(url, args.browser), daemon=True).start()

    app.run(host=host, port=port, debug=debug)
