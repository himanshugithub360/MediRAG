# 🩺 MediRAG — Dr. Insight, an AI Medical Chatbot

Dr. Insight is a medical question-answering chatbot built on **Retrieval-Augmented Generation (RAG)**. It answers health questions using passages retrieved from the *Gale Encyclopedia of Medicine*, rather than relying only on what the language model remembers. This makes answers more grounded and less likely to be made up.

> ⚠️ **Disclaimer:** This project is for learning and demonstration only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified doctor.

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Tech stack](#tech-stack)
4. [Project structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Setup (step by step)](#setup-step-by-step)
7. [Running the app](#running-the-app)
8. [Using the chatbot](#using-the-chatbot)
9. [Configuration reference](#configuration-reference)
10. [API reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [Customisation ideas](#customisation-ideas)

---

## Features

- 💬 Chat web interface (Flask + HTML/CSS/JS) with suggested starter questions
- 📚 Answers grounded in a medical encyclopedia via vector search (Pinecone)
- 🧠 Conversation memory: follow-ups like *"what are its treatments?"* are understood in context
- ⚡ Instant replies for greetings like "hi", "thanks", "bye" (no API call needed)
- 🩺 An empathetic "Dr. Insight" persona that recommends seeing a real doctor
- 🔧 LLM model selectable through an environment variable

## How it works

```
                ┌──────────────── one-time ingestion (store_index.py) ───────────────┐
 Data/*.pdf ──► PyPDFLoader ──► split into 300-char chunks ──► MiniLM embeddings ──► Pinecone index
                └────────────────────────────────────────────────────────────────────┘

                ┌──────────────── every chat message (app.py) ───────────────────────┐
 User question ──► MiniLM embedding ──► Pinecone: top 5 similar chunks
                                              │
                                              ▼
                     System prompt + retrieved chunks + chat history + question
                                              │
                                              ▼
                                      Mistral LLM ──► answer shown in the chat
                └────────────────────────────────────────────────────────────────────┘
```

1. **Ingestion (run once):** `store_index.py` reads the PDFs in `Data/` and splits them into small overlapping chunks. It turns each chunk into a 384-dimension vector with the `all-MiniLM-L6-v2` embedding model and uploads them to a Pinecone index named `medichatbot` (namespace `medical_data`).
2. **Chatting:** `app.py` embeds the user's question and fetches the 5 most similar chunks from Pinecone. It passes them, together with the recent conversation, to a Mistral chat model, which writes the final answer.

## Tech stack

| Layer            | Technology                                                   |
|------------------|--------------------------------------------------------------|
| Web server       | Flask                                                        |
| RAG framework    | LangChain (0.3.x)                                            |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (runs locally, CPU) |
| Vector database  | Pinecone (serverless, AWS `us-east-1`)                       |
| LLM              | Mistral AI (default: `open-mistral-nemo`)                    |
| PDF parsing      | pypdf via LangChain `PyPDFLoader`                            |

## Project structure

```
AI Medical-chatbot/
├── app.py               # Flask web app + RAG chain (main entry point)
├── store_index.py       # One-time script: PDFs → chunks → embeddings → Pinecone
├── src/
│   ├── helper.py        # PDF loading, text splitting, embedding model helpers
│   └── prompt.py        # Alternative/basic system prompt (not used by app.py)
├── templates/
│   └── chat.html        # Chat UI (HTML, CSS and JavaScript in one file)
├── static/              # Favicons, web manifest, style.css
├── Data/                # Source PDF(s) — Gale Encyclopedia of Medicine
├── research/
│   └── trials.ipynb     # Notebook used to experiment while building the project
├── requirements.txt     # Python dependencies
├── setup.py             # Makes `src` installable as a package (pip install -e .)
├── template.py          # Scaffolding script that created the initial folder layout
└── .env                 # Your API keys (NOT committed — you create this)
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10 – 3.12** | 3.12 is recommended and tested. Python 3.13/3.14 may not have wheels for every dependency yet. |
| **Git** | To clone the repository. |
| **Pinecone account** | Free tier works. Get an API key at <https://app.pinecone.io>. |
| **Mistral AI account** | Free tier works. Get an API key at <https://console.mistral.ai/api-keys>. |
| ~2 GB disk space | For PyTorch (CPU) and the embedding model. |
| Internet connection | To call Pinecone and Mistral, and download the embedding model on first run. |

No GPU is needed; everything runs on CPU.

---

## Setup (step by step)

The commands below are for **Windows** (Git Bash or PowerShell). macOS/Linux differences are noted where relevant.

### 1. Clone the repository

```bash
git clone https://github.com/himanshugithub360/MediRAG.git
```
```bash
cd MediRAG
```

### 2. Create a virtual environment

**Windows:**
```bash
py -3.12 -m venv .venv
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv
```

### 3. Activate it

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Git Bash:**
```bash
source .venv/Scripts/activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Activate it every time you open a new terminal to work on the project.

> If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or simply call `.venv\Scripts\python.exe` directly instead of activating.

### 4. Install dependencies

First install the **CPU-only** build of PyTorch (much smaller than the default GPU build):

```bash
python -m pip install --upgrade pip
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else (this also installs the local `src` package via `-e .`):

```bash
pip install -r requirements.txt
```

### 5. Create the `.env` file

Create a file called `.env` in the project root (same folder as `app.py`) with:

```env
PINECONE_API_KEY=your-pinecone-api-key
MISTRAL_API_KEY=your-mistral-api-key
FLASK_SECRET_KEY=any-long-random-string

# Optional
# MISTRAL_MODEL=open-mistral-nemo
# PORT=8080
```

- `FLASK_SECRET_KEY` signs the browser session cookie that identifies your conversation. If it's not set, a random key is generated on each start, so conversations reset when the server restarts. Any random string works; you can generate one with `python -c "import secrets; print(secrets.token_hex(32))"`.
- `.env` is listed in `.gitignore`. **Never commit your API keys.**

### 6. Build the Pinecone index (first time only)

> **Skip this step if your Pinecone account already has a `medichatbot` index with vectors in it.** You can check in the Pinecone console. Re-running it uploads the same chunks again.

Make sure the PDF is in `Data/`, then run:

```bash
python store_index.py
```

This will:
1. Load every PDF in `Data/`
2. Split them into chunks (~12,000 chunks for the included encyclopedia)
3. Create the `medichatbot` index (384 dimensions, cosine metric) if it does not exist
4. Embed and upload all chunks to the `medical_data` namespace

It takes several minutes, depending on your CPU and connection. At the end it prints the total vector count.

---

## Running the app

```bash
python app.py
```

When the server is ready, the chat opens automatically in your default browser at **<http://localhost:8080>**.

### Choosing the browser

| Command | Opens in |
|---------|----------|
| `python app.py` | Your system's default browser |
| `python app.py --browser chrome` | Google Chrome |
| `python app.py --browser edge` | Microsoft Edge |
| `python app.py --browser none` | Nothing. Open <http://localhost:8080> yourself in any browser (Chrome, Edge, Firefox, Brave, …) |

To make a choice permanent, add `OPEN_BROWSER=chrome` (or `edge`, `default`, `none`) to `.env`. A `--browser` flag on the command line overrides it. If the chosen browser isn't installed, the app opens your default browser instead. Chrome and Edge are found automatically on Windows, macOS and Linux.

### Notes

- The first start is slower because it downloads the embedding model (~90 MB) to `~/.cache/huggingface`. Later starts are quick.
- By default the server listens only on `127.0.0.1` (your own computer). To have it reload automatically when you edit Python files, set `FLASK_DEBUG=1` in `.env` (debug mode is always kept on `127.0.0.1` for safety).
- To stop the server, press `Ctrl + C`.
- To use another port, set `PORT=5000` in `.env`.

Running without activating the venv (Windows):
```bash
.venv/Scripts/python app.py
```

## Using the chatbot

- Type a medical question (for example, *"What are the symptoms of diabetes?"*) and press Enter, or click a suggested question.
- Follow-up questions work: the bot first rewrites a follow-up into a standalone question (e.g. *"what are its treatments?"* → *"what are the treatments for asthma?"*) before searching, and the last 8 messages are sent to the LLM as context.
- Messages that are *only* a greeting or thanks ("hi", "thanks!", "bye") are answered instantly without calling the LLM. A greeting followed by a question ("hey, what causes diabetes?") is answered normally.
- Chat history is kept in the server's memory (the browser cookie only holds a conversation ID). It is cleared when the server restarts. Clear cookies or open a private window to start a fresh conversation.

## Configuration reference

### Environment variables (`.env`)

| Variable            | Required | Default             | Purpose |
|---------------------|----------|---------------------|---------|
| `PINECONE_API_KEY`  | ✅       | —                   | Access to the Pinecone vector database |
| `MISTRAL_API_KEY`   | ✅       | —                   | Access to the Mistral LLM |
| `FLASK_SECRET_KEY`  | Recommended | random per start | Signs the session cookie |
| `MISTRAL_MODEL`     | ❌       | `open-mistral-nemo` | Which Mistral model answers questions |
| `PORT`              | ❌       | `8080`              | Port the web server listens on |
| `HOST`              | ❌       | `127.0.0.1`         | Address to listen on. Use `0.0.0.0` to allow other devices on your network (ignored while debug mode is on) |
| `FLASK_DEBUG`       | ❌       | off                 | `1` enables auto-reload and the debugger. Never enable it on a network-visible server |
| `OPEN_BROWSER`      | ❌       | `default`           | Browser to open on startup: `default`, `chrome`, `edge` or `none` |

**Choosing `MISTRAL_MODEL`:** your Mistral plan decides which models you can use. `open-mistral-nemo` works on the free tier. With a paid plan you can switch to stronger models such as `mistral-small-latest`, `mistral-medium-latest` or `mistral-large-latest` for better answers.

### Values in code you may want to tweak

| Setting | File | Default |
|---------|------|---------|
| Pinecone index name | `app.py`, `store_index.py` | `medichatbot` |
| Pinecone namespace | `app.py`, `store_index.py` | `medical_data` |
| Chunks retrieved per question (`k`) | `app.py` | `5` |
| LLM temperature | `app.py` | `0.2` |
| Chunk size / overlap | `src/helper.py` | `300` / `50` characters |
| Chat history length sent to the LLM | `app.py` (`MAX_HISTORY_MESSAGES`) | last 8 messages |
| Quick replies for greetings | `app.py` (`QUICK_RESPONSES`, `GREETING_ALIASES`) | hi, hello, thanks, bye, … |
| System prompt / persona | `app.py` (`system_prompt`) | "Dr. Insight" |

> If you change the embedding model, the vector dimension changes too. You must then recreate the Pinecone index with the new dimension and rerun `store_index.py`.

## API reference

### `GET /`
Returns the chat web page.

### `POST /chat`
Send a message and get the bot's reply.

**Request:**
```json
{ "message": "What causes migraines?" }
```

**Response:**
```json
{ "response": "Migraines are ..." }
```

Example with curl:
```bash
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d "{\"message\": \"What causes migraines?\"}"
```

---

## Troubleshooting

| Problem | Cause & fix |
|---------|-------------|
| `Missing environment variables: ...` on start | `.env` is missing, misnamed (e.g. `.env.txt`), or not in the project root. Check the file name and location. |
| Bot always replies *"I'm having trouble processing that request"* | The LLM call failed. Check the terminal running `app.py` for the full error. Common causes are listed in the next rows. |
| Error `403 ... not available in your subscription tier` | Your Mistral plan can't use the configured model. Set `MISTRAL_MODEL=open-mistral-nemo` in `.env`. |
| Error `429 Rate limit exceeded` | The Mistral free tier allows about 1 request per second, and has monthly limits per model. Wait a moment, send messages more slowly, or switch `MISTRAL_MODEL`. |
| Answers say there isn't enough information | The Pinecone index is empty or has a different name or namespace. Check the vector count in the Pinecone console, or run `store_index.py`. |
| `Index 'medichatbot' not found` | You haven't run `store_index.py` on this Pinecone account yet (step 6). |
| `ModuleNotFoundError` (e.g. `langchain_huggingface`, `src`) | The venv isn't activated, or dependencies aren't installed. Activate `.venv` and rerun `pip install -r requirements.txt`. |
| PyTorch download is huge / slow | You installed the default GPU build. Uninstall it and install the CPU build with the command in step 4. |
| Hugging Face "symlinks" warning on Windows | Harmless. To hide it, set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, or enable Windows Developer Mode. |
| `Address already in use` / port 8080 busy | Another program is using the port. Set `PORT=5000` in `.env`, or stop the other program. |
| Browser opens but shows "This site can't be reached" | The server stopped or is still starting. Check the terminal for errors, then refresh. Use `http://localhost:8080` (not `https://`). |
| No browser window opens | `OPEN_BROWSER=none` may be set in `.env`. Open <http://localhost:8080> manually, or run `python app.py --browser chrome`. |
| Install fails on Python 3.13/3.14 | Some packages lack wheels for very new Python versions. Use Python 3.12. |

## Customisation ideas

- **Add more knowledge:** drop more PDFs into `Data/` and rerun `store_index.py`.
- **Change the persona:** edit `system_prompt` in `app.py`.
- **Use a different LLM:** swap `ChatMistralAI` for another LangChain chat model (OpenAI, Anthropic, Groq, a local Ollama model, ...).
- **Show sources:** `rag_chain.invoke(...)` also returns `context` (the retrieved chunks). Return their page numbers to the UI to cite sources.
- **Deploy:** run behind a production WSGI server (e.g. `waitress` on Windows, `gunicorn` on Linux) (keep `FLASK_DEBUG` off).

---

## Author

**Himanshu Kumar** — [github.com/himanshugithub360](https://github.com/himanshugithub360)
