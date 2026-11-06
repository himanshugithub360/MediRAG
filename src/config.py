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

# Friendly titles for source files shown under answers. Files not listed here are
# shown by their file name.
SOURCE_TITLES = {
    "Gale Encyclopedia of Medicine. Vol. 1. 2nd ed.pdf": "Gale Encyclopedia of Medicine, Vol. 1",
    "OpenStax Pharmacology for Nurses.pdf": "OpenStax Pharmacology for Nurses",
}
