"""All settings in one place. Values marked "(env)" can be overridden in .env or as
environment variables (e.g. Hugging Face Space secrets/variables).

Used by app.py (chat), store_index.py (ingestion), fetch_sources.py, eval_retrieval.py and src/*.
"""
import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def _env(name, default):
    return os.environ.get(name, default)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


# ── API keys (env) ───────────────────────────────────────────────────────────
PINECONE_API_KEY = _env("PINECONE_API_KEY", "")
MISTRAL_API_KEY = _env("MISTRAL_API_KEY", "")
FLASK_SECRET_KEY = _env("FLASK_SECRET_KEY", "")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")                      # source PDFs + web/
INSTANCE_DIR = _env("INSTANCE_DIR", os.path.join(PROJECT_ROOT, "instance"))   # (env) runtime data
DB_PATH = _env("CHAT_DB_PATH", os.path.join(INSTANCE_DIR, "medirag.db"))     # (env) chat history
SECRET_KEY_PATH = os.path.join(INSTANCE_DIR, "secret_key")
# Local copy of every chunk, used for keyword (BM25) search. Written by store_index.py, or
# downloaded from Pinecone at startup when missing (e.g. on a fresh server).
CHUNKS_PATH = _env("CHUNKS_PATH", os.path.join(PROJECT_ROOT, "artifacts", "chunks.jsonl"))

# ── Pinecone ─────────────────────────────────────────────────────────────────
INDEX_NAME = "medichatbot"
# v2 = 1000-char chunks from all sources. The original 300-char index lives in the
# "medical_data" namespace and is only used as a baseline by eval_retrieval.py.
NAMESPACE = _env("PINECONE_NAMESPACE", "medical_v2")                       # (env)
LEGACY_NAMESPACE = "medical_data"
# Rebuild artifacts/chunks.jsonl from Pinecone if it's missing (needed on deployed servers)
CHUNKS_FROM_PINECONE = _env_bool("CHUNKS_FROM_PINECONE", True)              # (env)

# ── Chunking (rerun store_index.py after changing) ───────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Models ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim vectors, runs locally
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # runs locally
MISTRAL_MODEL = _env("MISTRAL_MODEL", "open-mistral-nemo")                  # (env) answers
MISTRAL_VISION_MODEL = _env("MISTRAL_VISION_MODEL", "pixtral-12b-2409")     # (env) photos/scans
LLM_TEMPERATURE = 0.2

# ── Retrieval ────────────────────────────────────────────────────────────────
CANDIDATES_PER_RETRIEVER = 20   # fetched from each of vector search and BM25
FINAL_K = 5                     # passages kept after reranking and sent to the LLM
VECTOR_WEIGHT = 0.6             # vs. BM25 (0.4) when fusing the two result lists
MAX_HISTORY_MESSAGES = 8        # recent chat messages sent to the LLM as context

# ── Safety ───────────────────────────────────────────────────────────────────
# Numbers shown for emergencies (defaults are for India)
EMERGENCY_NUMBER = _env("EMERGENCY_NUMBER", "112")                          # (env)
AMBULANCE_NUMBER = _env("AMBULANCE_NUMBER", "108")                          # (env)
MENTAL_HEALTH_HELPLINE = _env("MENTAL_HEALTH_HELPLINE", "14416")            # (env)
# If the best reranked passage scores below this, the bot declines to answer instead of
# guessing. Cross-encoder scores are logits; see the README for how this was set.
MIN_RELEVANCE_SCORE = float(_env("MIN_RELEVANCE_SCORE", "-1.0"))            # (env)

# ── Rate limits (Flask-Limiter syntax, per visitor IP) ──────────────────────
CHAT_RATE_LIMIT = _env("CHAT_RATE_LIMIT", "10 per minute;100 per day")      # (env)
UPLOAD_RATE_LIMIT = _env("UPLOAD_RATE_LIMIT", "5 per minute;30 per day")    # (env)
# Minimum gap between calls to Mistral across all users (free tier allows ~1 request/second)
MISTRAL_MIN_INTERVAL = float(_env("MISTRAL_MIN_INTERVAL", "1.1"))           # (env)

# ── Uploads ──────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "10"))                            # (env)
MAX_PDF_PAGES = 40
MAX_SCANNED_PAGES = 4           # scanned pages sent to the vision model (each is one API call)
MAX_UPLOAD_TEXT_CHARS = 60000

# ── Web server ───────────────────────────────────────────────────────────────
HOST = _env("HOST", "127.0.0.1")                                            # (env)
PORT = int(_env("PORT", "8080"))                                            # (env)
FLASK_DEBUG = _env_bool("FLASK_DEBUG")                                      # (env) dev server only
OPEN_BROWSER = _env("OPEN_BROWSER", "default").lower()                      # (env) dev server only
# Behind a reverse proxy (Hugging Face Spaces, Render, ...): trust X-Forwarded-* headers so
# rate limits apply per visitor instead of to the proxy's single IP address.
TRUST_PROXY = _env_bool("TRUST_PROXY")                                      # (env)
# Allow the session cookie inside a cross-site iframe (the Hugging Face Space page embeds
# the app from *.hf.space). Requires HTTPS.
CROSS_SITE_COOKIES = _env_bool("CROSS_SITE_COOKIES")                        # (env)

# ── Knowledge sources ────────────────────────────────────────────────────────
# Friendly titles for source files shown under answers. Files not listed here are
# shown by their file name.
SOURCE_TITLES = {
    "Gale Encyclopedia of Medicine. Vol. 1. 2nd ed.pdf": "Gale Encyclopedia of Medicine, Vol. 1",
    "OpenStax Pharmacology for Nurses.pdf": "OpenStax Pharmacology for Nurses",
}

# Knowledge collections users can filter by. Every chunk is tagged with one of these keys.
# PDFs are matched by file name (any Gale volume -> "encyclopedia"); other PDFs -> "other".
COLLECTIONS = {
    "encyclopedia": "Gale Encyclopedia of Medicine",
    "pharmacology": "OpenStax Pharmacology",
    "medlineplus": "MedlinePlus (NIH)",
    "who": "WHO fact sheets",
    "other": "Other PDFs",
}
WEB_SOURCES = {   # files in Data/web/ written by fetch_sources.py
    "medlineplus.jsonl": ("medlineplus", "MedlinePlus"),
    "who_factsheets.jsonl": ("who", "WHO fact sheet"),
}


def collection_for_file(file_name):
    name = file_name.lower()
    if "gale" in name or "encyclopedia" in name:
        return "encyclopedia"
    if "pharmacology" in name:
        return "pharmacology"
    return "other"
