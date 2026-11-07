# Dr. Insight (MediRAG) — production image.
# Works locally (docker run) and on Hugging Face Spaces (Docker SDK, free CPU tier).
#
#   docker build -t medirag .
#   docker run --env-file .env -p 7860:7860 medirag      -> http://localhost:7860
#
# API keys are passed at run time (--env-file / Space secrets), never baked into the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=7860

# Hugging Face Spaces runs containers as user id 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app
# The app writes chat history (instance/) and the keyword index (artifacts/) here at runtime
RUN chown user:user /app

# 1) Dependencies (cached unless requirements.txt changes). CPU-only PyTorch keeps the
#    image ~1.5 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^-e' requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

# 2) Bake the two local models into the image so containers start faster
USER user
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3) App code (see .dockerignore for what's left out: .env, Data/, chat history, ...)
COPY --chown=user . .

EXPOSE 7860

# One worker keeps a single copy of the models in memory; threads serve visitors in
# parallel. --preload loads the models (and downloads the keyword index from Pinecone on
# first start) before the port opens. Long timeout for streamed answers and uploads.
CMD gunicorn wsgi:app --preload --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 300 \
    --access-logfile - --error-logfile -
