<h1 align="center">WebChat</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/Vite-8.3+-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-4.0-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/uv-Astral-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="Astral UV" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-orange?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>
<h3 align="center">Multimodal Agentic RAG Web Conversational System</h3>
<p align="center">Chat with any public webpage, technical article, or documentation in real-time with visual diagram extraction, agentic multi-stage retrieval, multi-model fallback, and persistent conversation memory.</p>

---

## Overview

**WebChat** transforms static webpages and complex technical articles into interactive, contextual conversations. Powered by a cyclic multi-agent **LangGraph** orchestration pipeline, WebChat extracts text, code blocks, and **visual architectural diagrams/illustrations**, indexing them with dense embeddings and sparse keywords for ultra-precise retrieval with live streaming answers and citations.

---

## Features

- 🌐 **Chat with Any Webpage & Article**: Multi-stage scraping engine (Trafilatura, Jina Reader, BeautifulSoup4) with built-in anti-bot bypass for Medium articles, technical blogs, and documentation.
- 🖼️ **Multimodal Architectural Diagram Extraction**: Automatically parses, filters, and indexes architecture diagrams and figures, rendering them inline directly within chat answers.
- 🧠 **Agentic Multi-Agent Pipeline (LangGraph)**:
  - **Router Agent**: Detects user intent, categorizing queries into factual, complex, visual/diagram inquiries, or chitchat.
  - **Research Agent**: Performs sub-query parallel decomposition across hybrid dense semantic search (Qdrant / FAISS) and sparse keyword search (BM25).
  - **FlashRank Cross-Encoder Reranker**: Reranks top candidates to surface the most relevant context.
  - **Critic Agent (CRAG & Self-RAG)**: Automatically evaluates retrieval relevance, filters noise, prevents hallucinations, and re-queries or triggers web fallback if context is insufficient.
  - **Supervisor Agent**: Synthesizes verified responses with Markdown formatting, inline diagram embeds, and citations.
- 🔄 **Document-Preserved Multi-Session ("New Chat (Same Doc)")**: Switch between independent conversation topics on the active document without re-scraping or re-indexing, with lazy session initialization and persistence in `localStorage`.
- ⚡ **Multi-LLM Fallback & Cascading**: Seamless resilience between Google Gemini (`gemini-2.5-flash`, `gemini-3.6-flash`) and Groq (`llama-3.3-70b-versatile`) with automatic rate-limit backoff.
- 💾 **Persistent Semantic Caching & Memory**: Upstash Redis semantic cache for instant cached responses, PostgreSQL (Aiven) for conversation sessions and message history, and Memori Labs for cross-session user personalization.
- 📊 **50 Daily Queries Guest Quota**: Transparent device-based rate-limiting with a progress bar — no mandatory login or email capture walls.
- 🎨 **Dual Frontend Interfaces**:
  - **React 19 + Vite + Tailwind CSS 4**: Modern glassmorphic interface with Lucide icons, Framer Motion animations, real-time reasoning stepper, and suggested question chips.
  - **Streamlit Dashboard**: Analytical UI with full parameter control, interactive sidebar session history, and live query diagnostics.

---

## Project Structure

```
webchat/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph agents (Router, Research, Critic, Supervisor, Chat)
│   │   ├── api/             # FastAPI routers (chat, agent, user, frontend)
│   │   ├── cache/           # Redis semantic cache & vector store caching
│   │   ├── core/            # Configuration, logging, errors, security headers
│   │   ├── dtos/            # Pydantic request/response data schemas
│   │   ├── helpers/         # Streamlit & service launch utilities
│   │   ├── models/          # SQLAlchemy database models
│   │   ├── repositories/    # Database repository layer (sessions, messages, URLs)
│   │   └── services/        # RAG, scraping, LLM fallback, and memory services
│   ├── config/              # PostgreSQL database initialization
│   ├── main.py              # Application entry point (FastAPI server & Streamlit runner)
│   ├── pyproject.toml       # Backend dependencies managed by Astral UV
│   └── uv.lock              # Pinned UV dependency lockfile
├── frontend/                # Modern React 19 + Vite + Tailwind CSS 4 web app
│   ├── src/
│   │   ├── App.jsx          # Main chat interface with sidebar, navbar, and reasoning stepper
│   │   ├── App.css          # Glassmorphic layout styling
│   │   ├── index.css        # Tailwind CSS 4 directives
│   │   └── main.jsx         # React application root
│   ├── package.json         # Frontend dependencies and scripts
│   └── vite.config.js       # Vite build & plugin configuration
├── streamlit_app/           # Alternative Streamlit dashboard
│   ├── app.py               # Streamlit application entry point
│   ├── components/          # Chat, sidebar, and header UI components
│   └── styles.py            # Custom CSS styling for Streamlit
├── data/                    # Local storage, FAISS indices, and temporary cache
└── ARCHITECTURE.md          # In-depth architectural breakdown and state diagrams
```

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

1. **Python** `>= 3.11` (Python 3.11, 3.12, or 3.13)
2. **Astral `uv`** (fast Python package manager):
   - **Windows (PowerShell):**
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **macOS / Linux:**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
3. **Node.js** `>= 18.0.0` and **`npm`** (for the modern React frontend):
   - Check with `node -v` and `npm -v`

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/LaxmiNarayana31/webchat.git
cd webchat
```

### 2. Configure Environment Variables

Create your `.env` configuration inside the `backend/` directory from the provided template:

**Windows (PowerShell):**

```powershell
Copy-Item backend/.env.example backend/.env
```

**macOS / Linux:**

```bash
cp backend/.env.example backend/.env
```

Open `backend/.env` and provide your API keys:

```env
# Database Credentials (Aiven PostgreSQL or local Postgres)
DB_USER=avnadmin
DB_PASSWORD=your_db_password
DB_HOST=your_db_host.aivencloud.com
DB_PORT=5432
DB_NAME=defaultdb
SSL_MODE=require

# Google Gemini API Key (https://aistudio.google.com/api-keys)
GEMINI_API_KEY=your_gemini_api_key

# Groq API Key (https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key

# Upstash Redis Semantic Cache (https://console.upstash.com/redis)
REDIS_URL=rediss://default:YOUR_TOKEN@YOUR_HOST.upstash.io:6379

# Qdrant Vector Cloud (https://cloud.qdrant.io/)
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Cryptographic URL Hash Secret
URL_HASH_ALGORITHM=sha256
URL_HASH_SECRET=your_random_secret_string
```

> **Note:** If `QDRANT_URL` is omitted or unavailable, the backend automatically falls back to an in-memory **FAISS** index with zero configuration required.

---

### 3. Install Backend Dependencies & Run the API

From the root directory, navigate to `backend/` and sync dependencies using `uv`:

```bash
cd backend
uv sync
```

Start the FastAPI server with auto-reload:

```bash
uv run main.py --reload
```

- **API Endpoint:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

---

### 4. Run the React Frontend (Development)

In a separate terminal window, navigate to `frontend/`:

```bash
cd frontend
npm install
npm run dev
```

- **React App:** `http://localhost:5173`
- The React application connects directly to the FastAPI backend running on port `8000`.

---

### 5. Production Deployment (Single-Port Hosting)

You can build the React frontend into static assets that FastAPI serves automatically from `http://localhost:8000`:

```bash
# 1. Build the production React frontend bundle
cd frontend
npm run build

# 2. Run the FastAPI backend
cd ../backend
uv run main.py
```

The FastAPI server mounts `frontend/dist` and serves the full modern web app, documentation, and API endpoints from a single host (`http://localhost:8000`).

---

### 6. Alternative: Streamlit Dashboard

If you prefer the analytical Streamlit UI, you can launch it with either command:

**Option A (from `backend/`):**

```bash
cd backend
uv run main.py streamlit
```

**Option B (directly from `streamlit_app/`):**

```bash
cd streamlit_app
uv run streamlit run app.py
```

- **Streamlit Interface:** `http://localhost:8501`

---

## Environment Variables Reference

| Variable                      | Description                                                                     |   Required   | Source                                                              |
| :---------------------------- | :------------------------------------------------------------------------------ | :-----------: | :------------------------------------------------------------------ |
| `GEMINI_API_KEY`            | Primary LLM and dense embeddings (`text-embedding-004`, `gemini-2.5-flash`) | **Yes** | [Google AI Studio](https://aistudio.google.com/api-keys)             |
| `GROQ_API_KEY`              | High-speed LLM fallback provider (`llama-3.3-70b-versatile`)                  | **Yes** | [Groq Console](https://console.groq.com/keys)                        |
| `DB_USER` / `DB_PASSWORD` | PostgreSQL credentials for session and URL cache persistence                    | **Yes** | [Aiven Console](https://console.aiven.io/) / Local Postgres          |
| `DB_HOST` / `DB_PORT`     | PostgreSQL host and port (default:`5432`)                                     | **Yes** | PostgreSQL Provider                                                 |
| `DB_NAME`                   | Database name (e.g.`defaultdb`)                                               | **Yes** | PostgreSQL Provider                                                 |
| `SSL_MODE`                  | Database SSL mode (`require` or `prefer`)                                   |   Optional   | Defaults to`require`                                              |
| `REDIS_URL`                 | Upstash Redis connection string for fast semantic caching                       |   Optional   | [Upstash Console](https://console.upstash.com/)                      |
| `QDRANT_URL`                | Qdrant Cloud cluster endpoint for vector storage                                |   Optional   | [Qdrant Cloud](https://cloud.qdrant.io/) (FAISS fallback if omitted) |
| `QDRANT_API_KEY`            | Qdrant Cloud API key                                                            |   Optional   | Qdrant Cloud                                                        |
| `URL_HASH_SECRET`           | Secret key used for secure URL content hashing                                  |   Optional   | Any arbitrary string                                                |

---

## API Reference

| Endpoint                    |   Method   | Description                                                             |
| :-------------------------- | :--------: | :---------------------------------------------------------------------- |
| `/health`                 |  `GET`  | Service liveness and database connection status                         |
| `/api/scrape`             |  `POST`  | Scrapes target URL, extracts text & diagrams, and builds vector index   |
| `/api/chat`               |  `POST`  | Sends user query and returns complete synthesized response with sources |
| `/api/chat/stream`        |  `POST`  | SSE endpoint returning live token stream and reasoning steps            |
| `/api/user/sessions`      |  `GET`  | Retrieves all saved conversation sessions for the active client ID      |
| `/api/user/sessions/{id}` | `DELETE` | Deletes a conversation session and all associated turns                 |
| `/docs`                   |  `GET`  | Interactive OpenAPI Swagger specification                               |

---

## Verification & Testing

Run the automated test suites to ensure everything is working:

```bash
# Verify backend imports and dependencies
cd backend
uv run python -c "import sys; sys.path.insert(0, '..'); from backend.main import app; print('Backend OK!')"

# Verify Streamlit application compilation
cd ../streamlit_app
uv run python -m py_compile app.py styles.py components/chat.py components/sidebar.py

# Verify React frontend build
cd ../frontend
npm run build
```

---

## License

This project is licensed under the **MIT License**.
