# 🩺 MediRAG — Dr. Insight, an AI Medical Chatbot

Dr. Insight is a medical question-answering chatbot built on **Retrieval-Augmented Generation (RAG)**. It answers health questions using passages retrieved from trusted medical sources (the *Gale Encyclopedia of Medicine*, OpenStax *Pharmacology for Nurses*, **MedlinePlus** from the U.S. National Library of Medicine, and **WHO fact sheets**), rather than relying only on what the language model remembers. Every answer lists the sources it was based on.

You can also **upload a lab report, prescription or photo** and ask about it, chat in **Hindi and 9 other Indian languages**, **speak** your question and **hear** the answer, or run a guided **symptom checker**.

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
9. [Safety features](#safety-features)
10. [Configuration reference](#configuration-reference)
11. [API reference](#api-reference)
12. [Running the tests](#running-the-tests)
13. [Troubleshooting](#troubleshooting)
14. [Customisation ideas](#customisation-ideas)

---

## Features

- 📎 **Upload your own reports**: PDF lab reports and prescriptions (including scanned ones) or photos; ask *"which values are abnormal?"*
- 🇮🇳 **Hindi and other Indian languages**: ask in हिन्दी, বাংলা, मराठी, தமிழ், తెలుగు, ગુજરાતી, ಕನ್ನಡ, മലയാളം or ਪੰਜਾਬੀ; the bot searches in English and answers in your language
- 🎤 **Voice input and output**: speak your question and hear the answer read aloud (free, uses the browser's built-in speech)
- 🩺 **Symptom checker**: a guided form (symptoms, age, duration, severity, warning signs) that suggests possible causes, urgency and which doctor to see
- 📷 **Medical image questions**: a vision model describes a photographed rash or reads a photographed report, with strong disclaimers
- 🗂️ **Filter by source**: search all sources, or only the encyclopedia, pharmacology book, MedlinePlus or WHO
- 🖥️ **Modern dashboard**: sidebar of saved chats, welcome screen with suggested questions, works on desktop and phone
- ⚡ **Streaming answers**: text appears word by word, with a **Stop** button
- 📝 **Markdown rendering**: headings, bullet lists, **bold** and tables display properly
- 💾 **Saved chat history** (SQLite): reopen, search, rename or delete past conversations; they survive restarts
- 📋 **Copy** any answer, and rate it **👍 / 👎** to flag bad answers for review
- 🌗 **Dark / light theme** toggle (follows your system setting until you choose)
- 📚 Answers grounded in **13,800+ passages**: ~1,900 pages of medical books, 1,015 MedlinePlus health topics and 243 WHO fact sheets
- 📖 **Sources shown under every answer** (book + page, or a link to the MedlinePlus / WHO page)
- 🔎 **Hybrid search**: meaning-based vector search (Pinecone) + keyword search (BM25), so exact drug and disease names are always found
- 🎯 **Reranking**: a cross-encoder model picks the 5 best passages out of ~40 candidates
- 🚨 **Emergency detection**: chest pain, stroke signs, overdose and similar emergencies get emergency numbers (112 / 108) immediately, without waiting for the AI
- 💙 **Crisis support**: messages about suicide or self-harm get the Tele-MANAS helpline (14416) right away
- 💊 **No personal dosing or prescribing**: requests like *"how much paracetamol should I take?"* are politely declined
- 🙅 **Won't guess**: if the books have nothing relevant, the bot says so instead of making something up
- ⏳ **Rate limiting**: protects the app and keeps Mistral's free-tier limit from being hit
- 📏 **Built-in retrieval evaluation** (`eval_retrieval.py`) to measure changes with numbers
- 🧠 Conversation memory: follow-ups like *"what are its treatments?"* are understood in context
- ⚡ Instant replies for greetings like "hi", "thanks", "bye" (no API call needed)
- 🩺 An empathetic "Dr. Insight" persona that recommends seeing a real doctor
- 🔧 LLM model selectable through an environment variable

## How it works

```
 ── ingestion: store_index.py (run once, and again whenever Data/ changes) ─────────────────
 fetch_sources.py ─► Data/web/*.jsonl (MedlinePlus topics, WHO fact sheets)
 Data/*.pdf + Data/web/ ─► clean text ─► 1000-char chunks, tagged with source + collection ─┬─► MiniLM embeddings ─► Pinecone (medical_v2)
                                                                                            └─► artifacts/chunks.jsonl (keyword search)

 ── every chat message: app.py ──────────────────────────────────────────────────────────────
 question (any language) ─► safety keyword checks (English + Hindi) ─► emergency / crisis / dosage? → fixed reply
          ─► LLM: faithful English translation (re-checked for safety) + standalone English search question
          ─► vector search: top 20 ┐   (optionally limited to one collection)
          ─► BM25 keyword search: top 20 ┴─► fuse (Reciprocal Rank Fusion) ─► cross-encoder rerank ─► best 5 passages
          ─► + relevant parts of the user's uploaded files (if any)
          ─► Mistral LLM: answer written in the user's language, streamed word by word, with sources

 upload ─► PDF text (pypdf) or, for photos / scanned pages, the Pixtral vision model's transcription ─► saved with the chat
```

1. **Ingestion:** `fetch_sources.py` downloads MedlinePlus and WHO content into `Data/web/`. `store_index.py` reads that plus every PDF in `Data/`, removes page footers and line-break hyphenation, and splits the text into 1000-character chunks with 150 characters of overlap. Each chunk keeps its source, page number or web link, and a *collection* tag (`encyclopedia`, `pharmacology`, `medlineplus`, `who`) used by the source filter. Chunks are embedded with `all-MiniLM-L6-v2` (384 dimensions) and uploaded to the Pinecone index `medichatbot`, namespace `medical_v2`. A local copy is saved to `artifacts/chunks.jsonl` for keyword search.
2. **Retrieval:** for each question, `src/retrieval.py` runs two searches. **Vector search** finds passages with the same *meaning*; **BM25** finds passages containing the same *words*, which matters for names like "metformin" that embeddings can blur. The two result lists are merged, then a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) reads the question together with each candidate and keeps the 5 that best answer it.
3. **Answering:** `app.py` sends those 5 passages (each labelled with its book and page), the recent conversation and the question to a Mistral chat model. The answer is returned together with its sources.

## Tech stack

| Layer            | Technology                                                   |
|------------------|--------------------------------------------------------------|
| Web server       | Flask                                                        |
| RAG framework    | LangChain (0.3.x)                                            |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (runs locally, CPU) |
| Keyword search   | BM25 (`rank_bm25`), in memory                                |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally, CPU)   |
| Vector database  | Pinecone (serverless, AWS `us-east-1`)                       |
| LLM              | Mistral AI (default: `open-mistral-nemo`)                    |
| Vision model     | Mistral Pixtral (`pixtral-12b-2409`), for photos and scans   |
| Voice            | Browser Web Speech API (speech recognition + synthesis)     |
| Images           | Pillow (checks and shrinks uploaded photos)                  |
| PDF parsing      | pypdf via LangChain `PyPDFLoader`                            |
| Chat history     | SQLite (`instance/medirag.db`)                               |
| Safety & limits  | Keyword rules (`src/safety.py`), Flask-Limiter               |
| Frontend         | Plain HTML/CSS/JS, marked (Markdown), DOMPurify (sanitising) |

## Project structure

```
AI Medical-chatbot/
├── app.py               # Flask web app: RAG pipeline, streaming chat, conversation + feedback APIs
├── fetch_sources.py     # Downloads MedlinePlus topics + WHO fact sheets into Data/web/
├── store_index.py       # Builds the knowledge base: PDFs + web content → chunks → Pinecone + local copy
├── eval_retrieval.py    # Measures retrieval quality (Hit@5, MRR) for different setups
├── eval/
│   └── questions.json   # Test questions + keywords used by eval_retrieval.py
├── src/
│   ├── config.py        # Shared settings: index, namespace, chunk size, models, k values
│   ├── helper.py        # PDF loading + cleaning, chunking, embeddings, chunk file I/O
│   ├── retrieval.py     # Hybrid search (vector + BM25) and cross-encoder reranking
│   ├── safety.py        # Emergency, crisis and dosage detection + their responses
│   ├── storage.py       # SQLite: saved conversations, messages, uploads and 👍/👎 feedback
│   ├── documents.py     # Reading uploaded PDFs/photos (pypdf + vision model)
│   ├── languages.py     # Supported languages + script-based language detection
│   ├── launcher.py      # Opens the chat in the default browser, Chrome or Edge
│   └── prompt.py        # Alternative/basic system prompt (not used by app.py)
├── tests/
│   ├── test_safety.py   # 58 tests for the safety rules (English, Hindi, Hinglish)
│   ├── test_storage.py  # 6 tests for saved chats and feedback
│   └── test_languages.py # 3 tests for language detection
├── templates/
│   └── chat.html        # Dashboard UI (HTML, CSS and JavaScript in one file)
├── static/              # Favicons, web manifest
│   └── vendor/          # marked (Markdown) + DOMPurify (HTML sanitising), bundled for offline use
├── instance/            # medirag.db, uploads/ (users' files) + secret_key; created on first run (not committed)
├── Data/                # Source PDFs + web/ (downloaded content); not committed
├── artifacts/           # chunks.jsonl, generated by store_index.py (not committed)
├── research/
│   └── trials.ipynb     # Notebook used to experiment while building the project
├── requirements.txt     # Python dependencies
├── requirements-dev.txt # + pytest, for running the tests
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

- `FLASK_SECRET_KEY` signs the browser session cookie that links your browser to its saved chats. If it's not set, a random key is generated once and saved to `instance/secret_key`. Any random string works; you can generate one with `python -c "import secrets; print(secrets.token_hex(32))"`.
- `.env` is listed in `.gitignore`. **Never commit your API keys.**

### 6. Add the source PDFs

The PDFs live in `Data/`, which is **not** committed to Git (the files are large and the encyclopedia is copyrighted). Put these files in `Data/`:

| File name in `Data/` | Book | Where to get it |
|----------------------|------|-----------------|
| `Gale Encyclopedia of Medicine. Vol. 1. 2nd ed.pdf` | Gale Encyclopedia of Medicine, Vol. 1 (2nd ed.) | Your own copy (copyrighted) |
| `OpenStax Pharmacology for Nurses.pdf` | OpenStax *Pharmacology for Nurses* (~1,200 pages, 47 MB) | Free from OpenStax, see below |

Download the pharmacology book directly from OpenStax:

```bash
curl -L -o "Data/OpenStax Pharmacology for Nurses.pdf" https://assets.openstax.org/oscms-prodcms/media/documents/Pharmacology-WEB.pdf
```

You can add **any other medical PDFs** too: every `*.pdf` in `Data/` is indexed. To show a friendly book title under answers instead of the file name, add the file to `SOURCE_TITLES` in `src/config.py`.

> **More Gale volumes:** the encyclopedia has several volumes, but they are copyrighted, so this project doesn't download them. If you own them, put the PDFs in `Data/` (any file name containing "Gale" is put in the *encyclopedia* collection) and rebuild.

Then download the free web sources (about 5 minutes; WHO pages are fetched politely, one per second):

```bash
python fetch_sources.py
```

| Source | What | Licence |
|--------|------|---------|
| [MedlinePlus](https://medlineplus.gov/xml.html) (U.S. National Library of Medicine) | ~1,000 health-topic summaries, from the official daily XML file | U.S. government content; see [MedlinePlus usage terms](https://medlineplus.gov/about/using/usingcontent/) |
| [WHO fact sheets](https://www.who.int/news-room/fact-sheets) | ~240 fact sheets on diseases and health topics | CC BY-NC-SA 3.0 IGO (non-commercial use with attribution) |

Re-run it now and then to pick up updates, then rebuild the knowledge base.

### 7. Build the knowledge base

```bash
python store_index.py
```

This will:
1. Load and clean every PDF in `Data/` (removing page footers and words hyphenated across lines)
2. Add the MedlinePlus and WHO documents from `Data/web/`, and split everything into 1000-character chunks with 150 characters of overlap (13,828 chunks for the two books + web sources)
3. Save a copy of the chunks to `artifacts/chunks.jsonl`, used for keyword search
4. Create the `medichatbot` Pinecone index (384 dimensions, cosine metric) if it doesn't exist
5. Clear the `medical_v2` namespace, then embed and upload all chunks to it

It takes about 12 minutes on a typical laptop CPU (about 2.5 to read the PDFs, 8.5 to embed and upload). **Run it again whenever you add, remove or change PDFs.** It replaces the namespace's contents, so the index always matches `Data/`.

> The app needs **both** Pinecone and `artifacts/chunks.jsonl`. If the chunk file is missing (for example on a new computer), the app still works but skips keyword search. Run `store_index.py` once to restore it.

### 8. (Optional) Measure retrieval quality

```bash
python eval_retrieval.py
```

This asks the 16 test questions in `eval/questions.json` and checks whether a passage containing the expected keywords comes back in the top 5. It compares the original setup with each improvement. No LLM calls are made, so it is free. Add `--verbose` to see the top source for each question.

Results on the current knowledge base:

| Setup | Hit@5 | MRR | Seconds / question |
|-------|------:|----:|-------------------:|
| Original: 300-char chunks, encyclopedia only, vector search | 56% | 0.52 | 0.6 |
| 1000-char chunks, all 4 sources, vector search | 100% | 0.81 | 0.3 |
| + BM25 keyword search (hybrid) | 100% | 0.81 | 0.3 |
| **+ cross-encoder reranker (what the app uses)** | **100%** | **0.87** | 1.6 |

(With only the two books, the app's setup scored 100% / 0.90. Adding ~6,000 web passages gives more competing candidates, so the first relevant passage is slightly less often ranked #1.)

- **Hit@5**: share of questions where a relevant passage was in the top 5.
- **MRR** (mean reciprocal rank): 1.0 means the relevant passage was always ranked first; 0.5 means it was second on average.
- Keyword matching is a rough automatic judge, so use these numbers to compare setups rather than as absolute accuracy. The old setup only had the encyclopedia, so it can't answer the drug questions at all.

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

- Type a medical question (for example, *"What are the symptoms of diabetes?"*) and press **Enter**, or click a suggested question. **Shift + Enter** adds a new line.
- The answer streams in word by word. Click the **■ Stop** button (where the send button was) to stop it early; what was written so far is kept.
- Follow-up questions work: the bot first rewrites a follow-up into a standalone question (e.g. *"what are its treatments?"* → *"what are the treatments for asthma?"*) before searching, and the last 8 messages are sent to the LLM as context.
- Messages that are *only* a greeting or thanks ("hi", "thanks!", "bye") are answered instantly without calling the LLM. A greeting followed by a question ("hey, what causes diabetes?") is answered normally.
- Under each answer:
  - **📖 N sources**: click to see the book and page each answer came from
  - **Copy**: copies the answer (as Markdown) to the clipboard
  - **👍 / 👎**: rate the answer. Click again to undo
- **New chat** (top of the sidebar) starts a fresh conversation. Earlier chats stay in the sidebar, grouped into *Today*, *Yesterday*, *Previous 7 days* and so on. Use the search box to filter them by title, and hover a chat to **rename ✏️** or **delete 🗑️** it.
- Each chat's address includes its id (`?c=…`), so refreshing the page keeps you in the same chat.
- **Light / dark mode**: button at the bottom of the sidebar. Your choice is remembered in this browser.
- On a phone, tap **☰** (top left) to open the sidebar.

### Asking in Hindi or another Indian language

Pick a language from the **🌐 language menu** at the top, or leave it on **Auto language**: questions typed in an Indian script (e.g. *"मधुमेह के लक्षण क्या हैं?"*) are detected automatically. For Hindi typed in English letters (*"mujhe bukhar hai"*), choose **हिन्दी** in the menu.

Behind the scenes the question is translated to English to search the (English) sources, and the answer is written in your language. Safety checks run on both the original text and the English translation, and common Hindi/Hinglish emergency phrases (*"सीने में दर्द हो रहा है"*, *"saans nahi aa rahi"*) are recognised directly.

### Voice: speak and listen

- Click **🎤** in the message box and speak. The question is sent when you stop talking, and **the answer is read aloud automatically**. Choose your language in the 🌐 menu first; speech recognition uses it (Auto = English).
- Click **🔊** under any answer to hear it; click again to stop.
- Voice input works in **Chrome and Edge** (the mic button is hidden in browsers without speech recognition). The browser asks for microphone permission the first time.
- Read-aloud uses the voices installed on your device. Windows usually has English (India) voices; for **Hindi read-aloud use Microsoft Edge** (it includes online Hindi voices) or add a Hindi voice in Windows *Settings → Time & language → Speech*.

### Uploading a report, prescription or photo

Click **📎** in the message box, or drag a file onto the chat. PDF, PNG, JPG and WebP up to 10 MB are accepted.

- **Text PDFs** (most lab reports from labs/hospitals) are read directly.
- **Scanned PDFs and photos** are read by Mistral's **Pixtral** vision model: it transcribes text (test values, units, reference ranges) into a table, or describes what is visible in a photo of skin or a wound **without naming a diagnosis**.
- Then ask anything: *"Explain my results in simple words"*, *"Which values are outside the normal range?"*, *"What does HbA1c measure?"*. The answer uses your file plus the medical sources, and lists the file as a source.
- Image answers always carry a warning that AI image reading can be wrong and is not a diagnosis.

Files are stored in `instance/uploads/` on the computer running the app and are **deleted together with their chat**.

### Symptom checker

Click **Symptom checker** (top right, or on the welcome screen). Fill in your symptoms (quick buttons for common ones), age, sex, how long, how bad (1–10), existing conditions, current medicines, and tick any **warning signs**. You'll get:

- **Possible causes** (clearly marked as not a diagnosis)
- **How urgent it is**: emergency now / see a doctor today / in the next few days / usually fine at home
- **Which doctor to see** (e.g. general physician, ENT specialist, dermatologist)
- **What you can do meanwhile** and **when to get urgent help**

Ticking a warning sign (chest pain, breathing difficulty, stroke signs, heavy bleeding, seizure, stiff neck with fever, thoughts of self-harm) skips the AI and shows the emergency or crisis card immediately.

### Filtering by source

The **📖 source menu** at the top limits answers to one collection: *Gale Encyclopedia*, *OpenStax Pharmacology*, *MedlinePlus (NIH)* or *WHO fact sheets*. *All sources* is the default. Your choice is remembered in this browser.

### Where chats are saved

Chats are stored in a SQLite database at `instance/medirag.db`, so they survive server restarts. Each browser gets an anonymous id in a signed cookie and only sees its own chats; there are no accounts. Clearing the browser's cookies starts a fresh, empty history (the old chats stay in the database).

> If `FLASK_SECRET_KEY` isn't set in `.env`, a random key is generated once and saved to `instance/secret_key`, so cookies stay valid across restarts.

### Reviewing 👎 feedback

Open <http://localhost:8080/api/feedback/summary> **on the computer running the app** (it's blocked for other devices). It shows the 👍 and 👎 totals and the 20 most recent 👎 answers together with the question that was asked, which makes weak answers easy to find and fix.

## Safety features

Every message goes through safety checks in [`src/safety.py`](src/safety.py) **before** the AI sees it. When a check triggers, the AI is skipped and a fixed, reviewed reply is shown instead, so it's instant and works even if Mistral is down or rate-limited.

| Situation | Example | What happens |
|-----------|---------|--------------|
| 🚨 **Emergency happening now** | *"My dad is having chest pain"*, *"my son swallowed bleach"*, *"her face is drooping"* | Red card: call **112** (emergency) or **108** (ambulance), plus short first-aid steps for that situation |
| 💙 **Suicidal thoughts / self-harm** | *"I want to end my life"*, *"my friend is suicidal"* | Blue card with **Tele-MANAS 14416** (free, 24×7) and **112** |
| 💊 **Personal dose or prescription** | *"How much paracetamol should I take?"*, *"Which medicine should I take for a cold?"* | Polite refusal; suggests a doctor or pharmacist and offers general information instead |
| ℹ️ **General question about an emergency topic** | *"What are the signs of a heart attack?"* | Answered normally, with an orange *"if this is happening now, call 112"* notice |
| 🙅 **Nothing relevant in the books** | *"What is long COVID?"*, *"What's the capital of France?"* | *"I couldn't find reliable information…"* instead of a guess |

On phones, the helpline numbers are tappable links that start a call.

**How "happening now" is told apart from a general question:** a symptom alone isn't enough. The message must also describe someone experiencing it (*"I'm…"*, *"my mother has…"*, *"her throat is swelling"*) or contain an urgent phrase (*"right now"*, *"help me"*, *"what should I do"*). Mentions of the past, prevention or risk (*"I had a stroke last year"*) are not treated as emergencies. Ambiguous messages lean towards showing emergency information.

**How "nothing relevant" is decided:** the reranker scores how well the best passage answers the question. Real medical questions scored between **+1.3 and +8.8**, off-topic ones between **−10.9 and −8.4**, and medical topics the books don't cover (long COVID, Zika, monkeypox) between **−9.7 and −2.4**. The cut-off is **−1.0** (`MIN_RELEVANCE_SCORE`).

**Also built into the AI's instructions:** never give specific doses, never tell the user which medicine to take, start or stop, and never diagnose.

**Rate limiting:** each visitor can send **10 messages per minute and 100 per day** (`CHAT_RATE_LIMIT`). Emergency and crisis messages are **never** rate-limited. Separately, calls to Mistral are spaced at least 1.1 seconds apart across all users (`MISTRAL_MIN_INTERVAL`), which keeps the free tier's ~1 request/second limit from being hit.

> ⚠️ These are keyword rules: fast, free and predictable, but not perfect. They can miss unusual wording or flag a harmless message. They support, but do not replace, professional care. The emergency numbers default to India; change them in `.env` for other countries.

## Configuration reference

### Environment variables (`.env`)

| Variable            | Required | Default             | Purpose |
|---------------------|----------|---------------------|---------|
| `PINECONE_API_KEY`  | ✅       | —                   | Access to the Pinecone vector database |
| `MISTRAL_API_KEY`   | ✅       | —                   | Access to the Mistral LLM |
| `FLASK_SECRET_KEY`  | Recommended | random, saved in `instance/secret_key` | Signs the session cookie that links a browser to its saved chats |
| `MISTRAL_MODEL`     | ❌       | `open-mistral-nemo` | Which Mistral model answers questions |
| `PORT`              | ❌       | `8080`              | Port the web server listens on |
| `HOST`              | ❌       | `127.0.0.1`         | Address to listen on. Use `0.0.0.0` to allow other devices on your network (ignored while debug mode is on) |
| `FLASK_DEBUG`       | ❌       | off                 | `1` enables auto-reload and the debugger. Never enable it on a network-visible server |
| `OPEN_BROWSER`      | ❌       | `default`           | Browser to open on startup: `default`, `chrome`, `edge` or `none` |
| `EMERGENCY_NUMBER`  | ❌       | `112`               | Emergency number shown in emergency and crisis replies |
| `AMBULANCE_NUMBER`  | ❌       | `108`               | Ambulance number shown in emergency replies |
| `MENTAL_HEALTH_HELPLINE` | ❌  | `14416`             | Mental-health helpline (Tele-MANAS) shown in crisis replies |
| `MIN_RELEVANCE_SCORE` | ❌     | `-1.0`              | Below this reranker score the bot declines to answer. Lower = answers more, but guesses more |
| `CHAT_RATE_LIMIT`   | ❌       | `10 per minute;100 per day` | Messages allowed per visitor (Flask-Limiter syntax) |
| `MISTRAL_MIN_INTERVAL` | ❌    | `1.1`               | Minimum seconds between Mistral calls across all users |
| `MISTRAL_VISION_MODEL` | ❌    | `pixtral-12b-2409`  | Model that reads uploaded photos and scanned PDFs |
| `UPLOAD_RATE_LIMIT` | ❌       | `5 per minute;30 per day` | Uploads allowed per visitor (each image uses a vision-model call) |

**Choosing `MISTRAL_MODEL`:** your Mistral plan decides which models you can use. `open-mistral-nemo` works on the free tier. With a paid plan you can switch to stronger models such as `mistral-small-latest`, `mistral-medium-latest` or `mistral-large-latest` for better answers.

### Values in code you may want to tweak

| Setting | File | Default |
|---------|------|---------|
| Pinecone index name | `src/config.py` (`INDEX_NAME`) | `medichatbot` |
| Pinecone namespace | `src/config.py` (`NAMESPACE`), or `PINECONE_NAMESPACE` in `.env` | `medical_v2` |
| Candidates per search / passages sent to the LLM | `src/config.py` (`CANDIDATES_PER_RETRIEVER`, `FINAL_K`) | `20` / `5` |
| Vector vs. keyword weight when merging results | `src/config.py` (`VECTOR_WEIGHT`) | `0.6` / `0.4` |
| Reranker model | `src/config.py` (`RERANKER_MODEL`) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Book titles shown in sources | `src/config.py` (`SOURCE_TITLES`) | Gale, OpenStax Pharmacology |
| LLM temperature | `app.py` | `0.2` |
| Chunk size / overlap (rerun `store_index.py` after changing) | `src/config.py` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) | `1000` / `150` characters |
| Chat history length sent to the LLM | `app.py` (`MAX_HISTORY_MESSAGES`) | last 8 messages |
| Quick replies for greetings | `app.py` (`QUICK_RESPONSES`, `GREETING_ALIASES`) | hi, hello, thanks, bye, … |
| System prompt / persona | `app.py` (`system_prompt`) | "Dr. Insight" |

> If you change the embedding model, the vector dimension changes too. You must then recreate the Pinecone index with the new dimension and rerun `store_index.py`.

## API reference

### `GET /`
Returns the chat web page.

All endpoints use the anonymous session cookie to know whose chats are whose.

| Method & path | Purpose |
|---------------|---------|
| `POST /api/chat/stream` | Send a message; the answer streams back as [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (used by the web page) |
| `POST /chat` | Send a message; get the whole answer as one JSON response (handy for scripts) |
| `GET /api/conversations` | List your saved chats, newest first |
| `GET /api/conversations/<id>` | One chat with all its messages |
| `PATCH /api/conversations/<id>` | Rename a chat: `{"title": "..."}` |
| `DELETE /api/conversations/<id>` | Delete a chat and its messages |
| `POST /api/messages/<id>/feedback` | Rate an answer: `{"value": 1}` (👍), `-1` (👎) or `null` (clear) |
| `POST /api/symptom-check/stream` | Guided symptom check (streams like `/api/chat/stream`). Body: `symptoms` (required), `age`, `sex`, `duration`, `severity`, `conditions`, `medicines`, `red_flags` (list) |
| `POST /api/uploads` | Upload a PDF or image (multipart form: `file`, optional `conversation_id`, `language`) |
| `GET /api/attachments/<id>/file` | Download/view an uploaded file (only its owner) |
| `GET /api/feedback/summary` | Feedback totals + recent 👎 answers (only from the computer running the app) |

Both chat endpoints take `{"message": "...", "conversation_id": "...", "language": "auto", "collections": []}`. Leave out `conversation_id` (or pass an unknown one) to start a new chat. `language` is `auto` or a code (`en`, `hi`, `bn`, `mr`, `ta`, `te`, `gu`, `kn`, `ml`, `pa`); `collections` can limit the search to any of `encyclopedia`, `pharmacology`, `medlineplus`, `who`.

### `POST /api/chat/stream`

Events, in order:

| Event | Data | Meaning |
|-------|------|---------|
| `conversation` | `{"id", "title"}` | The chat this message belongs to (new or existing) |
| `token` | `{"text"}` | The next piece of the answer; append them in order |
| `done` | `{"message_id", "content", "sources", "notice", "kind"}` | The complete saved answer |
| `error` | `{"text"}` | Something went wrong; `text` is a message to show |

### `POST /chat`

**Request:**
```json
{ "message": "What causes migraines?" }
```

**Response:**
```json
{
  "conversation_id": "5f3f5a1ae87e40dc8aca173aa6ab9ff0",
  "message_id": 12,
  "response": "Migraines are ...",
  "sources": [
    { "title": "Gale Encyclopedia of Medicine, Vol. 1", "page": "412" },
    { "title": "OpenStax Pharmacology for Nurses", "page": "301" }
  ]
}
```
| Field | When present | Meaning |
|-------|--------------|---------|
| `response` | always | The reply text (may contain `**bold**` and line breaks) |
| `sources` | normal answers | Book and page of each passage the answer was based on, best match first |
| `notice` | general questions about emergency topics | Extra *"if this is happening now, call…"* note to show with the answer |
| `kind` | safety replies | `emergency`, `crisis`, `dosage`, `low_confidence` or `rate_limited` |

When the rate limit is exceeded, the status code is **429** and `kind` is `rate_limited`.

Example with curl:
```bash
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d "{\"message\": \"What causes migraines?\"}"
```

---

## Running the tests

There are 67 automated tests:

- **58 for the safety rules**: emergencies, crises, dosage requests, general questions that should only get a notice, normal questions that must pass through untouched, and Hindi/Hinglish emergency phrases.
- **6 for saved chats**: titles, privacy between users, message order, feedback rules and deletion.
- **3 for language detection**: the 8 Indian scripts, English/Hinglish, and explicit language choice.

They need no API keys and run in about a second:

```bash
pip install -r requirements-dev.txt
```
```bash
python -m pytest -q
```

When you change a rule in `src/safety.py`, add the message that prompted the change to `tests/test_safety.py`.

## Troubleshooting

| Problem | Cause & fix |
|---------|-------------|
| `Missing environment variables: ...` on start | `.env` is missing, misnamed (e.g. `.env.txt`), or not in the project root. Check the file name and location. |
| Bot always replies *"I'm having trouble processing that request"* | The LLM call failed. Check the terminal running `app.py` for the full error. Common causes are listed in the next rows. |
| Error `403 ... not available in your subscription tier` | Your Mistral plan can't use the configured model. Set `MISTRAL_MODEL=open-mistral-nemo` in `.env`. |
| Error `429 Rate limit exceeded` | The Mistral free tier allows about 1 request per second, and has monthly limits per model. Wait a moment, send messages more slowly, or switch `MISTRAL_MODEL`. |
| Answers say there isn't enough information | The Pinecone index is empty or has a different name or namespace. Check the vector count in the Pinecone console, or run `store_index.py`. |
| `Index 'medichatbot' not found` | You haven't run `store_index.py` on this Pinecone account yet (step 7). |
| Log says `chunks.jsonl not found; keyword search disabled` | Run `python store_index.py` to create `artifacts/chunks.jsonl`. |
| `No PDF text found in .../Data` | Put the PDFs in `Data/` first (step 6). |
| `not enough memory` while indexing or chatting | Indexing and the app each load AI models. Close other heavy programs, and don't run `store_index.py` while the app is running. |
| `ModuleNotFoundError` (e.g. `langchain_huggingface`, `src`) | The venv isn't activated, or dependencies aren't installed. Activate `.venv` and rerun `pip install -r requirements.txt`. |
| PyTorch download is huge / slow | You installed the default GPU build. Uninstall it and install the CPU build with the command in step 4. |
| Hugging Face "symlinks" warning on Windows | Harmless. To hide it, set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, or enable Windows Developer Mode. |
| `Address already in use` / port 8080 busy | Another program is using the port. Set `PORT=5000` in `.env`, or stop the other program. |
| Browser opens but shows "This site can't be reached" | The server stopped or is still starting. Check the terminal for errors, then refresh. Use `http://localhost:8080` (not `https://`). |
| No browser window opens | `OPEN_BROWSER=none` may be set in `.env`. Open <http://localhost:8080> manually, or run `python app.py --browser chrome`. |
| Chat history disappeared | The browser's cookies were cleared, or you're in a private window or another browser. Chats are tied to the browser's cookie. |
| Answer appears all at once instead of streaming | A proxy or antivirus may be buffering the response. Streaming works when you open the app directly at `http://localhost:8080`. |
| *"You're sending messages too quickly"* | You hit the rate limit (10 per minute by default). Wait a minute, or raise `CHAT_RATE_LIMIT` in `.env`. |
| A normal question gets *"I couldn't find reliable information"* | The books may not cover it, or the wording is vague. Try naming the condition or medicine. To make the bot less strict, lower `MIN_RELEVANCE_SCORE` (e.g. `-3`). |
| A harmless message got an emergency or dosage reply | The keyword rules erred on the side of caution. Rephrase it as a general question, e.g. *"What are the signs of…"*. |
| No 🎤 button | Your browser doesn't support speech recognition (e.g. Firefox). Use Chrome or Edge. |
| *"Microphone access was blocked"* | Allow the microphone for `localhost` in the browser's site settings (padlock icon in the address bar). |
| Hindi answers are read aloud badly or not at all | No Hindi voice is installed. Use Microsoft Edge, or add a Hindi voice in Windows *Settings → Time & language → Speech*. |
| Hinglish (*"mujhe bukhar hai"*) is answered in English | Auto-detection only recognises Indian scripts. Choose **हिन्दी** in the language menu. |
| *"Couldn't find any readable text in that PDF"* | The PDF is scanned and the vision model couldn't read it. Upload a clearer photo of the page instead. |
| Uploading a photo gives a 429 / rate-limit message | Image reading uses a Mistral vision call; the free tier is limited. Wait a minute. |
| MedlinePlus / WHO sources never appear | Run `python fetch_sources.py`, then `python store_index.py`. |
| Install fails on Python 3.13/3.14 | Some packages lack wheels for very new Python versions. Use Python 3.12. |

## Customisation ideas

- **Add more knowledge:** drop more PDFs into `Data/` and rerun `store_index.py`.
- **Change the persona:** edit `system_prompt` in `app.py`.
- **Use a different LLM:** swap `ChatMistralAI` for another LangChain chat model (OpenAI, Anthropic, Groq, a local Ollama model, ...).
- **Tune retrieval:** change chunk size, `FINAL_K` or `VECTOR_WEIGHT` in `src/config.py`, rebuild, then rerun `eval_retrieval.py` to see whether it helped. Add your own questions to `eval/questions.json`.
- **Deploy:** run behind a production WSGI server (e.g. `waitress` on Windows, `gunicorn` on Linux) (keep `FLASK_DEBUG` off).

---

## Author

**Himanshu Kumar** — [github.com/himanshugithub360](https://github.com/himanshugithub360)
