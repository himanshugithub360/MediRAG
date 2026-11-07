"""Settings shared by store_index.py (ingestion), app.py (chat) and eval_retrieval.py."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source documents
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Pinecone
INDEX_NAME = "medichatbot"
# v2 = 1000-char chunks from all PDFs in Data/. The original 300-char index lives in
# the "medical_data" namespace and is only used as a baseline by eval_retrieval.py.
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "medical_v2")
LEGACY_NAMESPACE = "medical_data"

# Chunking. Bigger chunks keep a whole paragraph (e.g. a drug's side effects) together.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Local copy of every chunk, used for keyword (BM25) search. Written by store_index.py.
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "chunks.jsonl")

# Models (all run locally on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim vectors
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
CANDIDATES_PER_RETRIEVER = 20   # fetched from each of vector search and BM25
FINAL_K = 5                     # passages kept after reranking and sent to the LLM
VECTOR_WEIGHT = 0.6             # vs. BM25 (0.4) when fusing the two result lists

# Safety: numbers shown for emergencies (defaults are for India; override in .env)
EMERGENCY_NUMBER = os.environ.get("EMERGENCY_NUMBER", "112")
AMBULANCE_NUMBER = os.environ.get("AMBULANCE_NUMBER", "108")
MENTAL_HEALTH_HELPLINE = os.environ.get("MENTAL_HEALTH_HELPLINE", "14416")

# Safety: if the best reranked passage scores below this, the bot declines to answer
# instead of guessing. Cross-encoder scores are logits; see the README for how this was set.
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "-1.0"))

# Rate limiting (Flask-Limiter syntax). Per visitor IP address.
CHAT_RATE_LIMIT = os.environ.get("CHAT_RATE_LIMIT", "10 per minute;100 per day")
# Minimum gap between calls to Mistral across all users (free tier allows ~1 request/second)
MISTRAL_MIN_INTERVAL = float(os.environ.get("MISTRAL_MIN_INTERVAL", "1.1"))

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

