"""Build the knowledge base: every PDF in Data/ -> cleaned chunks -> Pinecone + local BM25 copy.

Run it again whenever you add, remove or change PDFs in Data/. It replaces the
contents of the Pinecone namespace, so the index always matches the folder.
"""
from src.config import (CHUNKS_PATH, DATA_DIR, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE)
from src.helper import (chunk_id, download_hugging_face_embeddings, load_pdf_file, save_chunks,
                        text_split)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from collections import Counter
from dotenv import load_dotenv
import os
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise SystemExit('PINECONE_API_KEY is not set. Add it to your .env file.')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

BATCH_SIZE = 200

# Load and process documents
print(f"Loading PDF documents from {DATA_DIR} ...")
extracted_data = load_pdf_file(data=DATA_DIR)
if not extracted_data:
    raise SystemExit(f"No PDF text found in {DATA_DIR}. Put your PDF files there first.")
for source, pages in Counter(d.metadata["source"] for d in extracted_data).items():
    print(f"  {source}: {pages} pages")

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
ids = [chunk_id(doc, i) for i, doc in enumerate(text_chunks)]
print(f"Created {len(text_chunks)} text chunks")

# Local copy for keyword (BM25) search in the app
save_chunks(text_chunks, ids, CHUNKS_PATH)
print(f"Saved chunks for keyword search to {CHUNKS_PATH}")

print(f"Initializing embeddings ({EMBEDDING_MODEL})...")
embeddings = download_hugging_face_embeddings()

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Matches all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index initialization...")
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(5)
    print("Index is ready!")

index = pc.Index(INDEX_NAME)

# Start the namespace from scratch so removed/changed PDFs don't leave stale vectors behind
if NAMESPACE in index.describe_index_stats().get("namespaces", {}):
    print(f"Clearing existing vectors in namespace '{NAMESPACE}'...")
    index.delete(delete_all=True, namespace=NAMESPACE)

vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)

print(f"Embedding and uploading to namespace '{NAMESPACE}' (this takes several minutes on CPU)...")
start = time.time()
for i in range(0, len(text_chunks), BATCH_SIZE):
    vector_store.add_documents(text_chunks[i:i + BATCH_SIZE], ids=ids[i:i + BATCH_SIZE])
    done = min(i + BATCH_SIZE, len(text_chunks))
    print(f"  {done}/{len(text_chunks)} chunks uploaded ({time.time() - start:.0f}s)", flush=True)

# Pinecone's stats are eventually consistent; wait briefly for the count to catch up
for _ in range(12):
    count = index.describe_index_stats().get("namespaces", {}).get(NAMESPACE, {}).get("vector_count", 0)
    if count >= len(text_chunks):
        break
    time.sleep(5)

print(f"\nDone! {count} vectors in namespace '{NAMESPACE}' of index '{INDEX_NAME}'.")
