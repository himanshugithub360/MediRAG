import hashlib
import json
import os
import re

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, SOURCE_TITLES

# Lines repeated on every page that only add noise to search results
_BOILERPLATE = re.compile(r"^\s*Access for free at openstax\.org\s*$", re.MULTILINE)


def clean_page_text(text):
    text = _BOILERPLATE.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single line breaks -> spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# Extract data from pdf files
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    cleaned = []
    for doc in documents:
        text = clean_page_text(doc.page_content)
        if not text:
            continue
        file_name = os.path.basename(doc.metadata.get("source", ""))
        page = doc.metadata.get("page_label") or str(doc.metadata.get("page", 0) + 1)
        cleaned.append(Document(
            page_content=text,
            metadata={
                "source": SOURCE_TITLES.get(file_name, os.path.splitext(file_name)[0]),
                "file": file_name,
                "page": str(page),
            },
        ))
    return cleaned


# Split the data into text chunks
def text_split(extracted_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(extracted_data)


def chunk_id(doc, position):
    """Stable id, so re-running ingestion overwrites vectors instead of duplicating them."""
    key = f"{doc.metadata.get('file')}|{doc.metadata.get('page')}|{position}|{doc.page_content[:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# Download the embeddings from Hugging Face
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def save_chunks(chunks, ids, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc, doc_id in zip(chunks, ids):
            f.write(json.dumps({"id": doc_id, "text": doc.page_content, "metadata": doc.metadata},
                               ensure_ascii=False) + "\n")


def load_chunks(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata=row["metadata"], id=row["id"]))
    return docs
