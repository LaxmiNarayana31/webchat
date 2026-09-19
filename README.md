<h1 align="center">WebChat</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/uv-Astral-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="Astral UV" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-orange?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>
<h3 align="center">Chat with any webpage in real time</h3>
<p align="center">Paste any public link like technical docs, blog posts, news, or articles and ask questions with streaming answers and citations.</p>

---

Under the hood, WebChat uses LangGraph for agentic retrieval, hybrid vector + keyword search, multi-model fallback across Gemini and Groq, and persistent memory.

## Features

- **Chat with any URL**: Scrapes and parses webpage content in real time. Includes fallbacks for JavaScript-rendered sites, Medium articles, and Wayback Machine archives.
- **Domain crawling**: Breadth-first crawler to index and search across entire documentation websites.
- **Hybrid search & reranking**: Combines dense semantic search (Qdrant) and sparse keyword search (BM25) with FlashRank reranking.
- **Agentic RAG**: Validates retrieved context, rewrites unclear queries, and falls back to web search (DuckDuckGo/Wikipedia) when needed.
- **Multi-LLM cascade**: Automatic fallback between Google Gemini and Groq models with rate limit handling and exponential backoff.
- **User memory**: Tracks user preferences and conversation history across sessions with Memori Labs and PostgreSQL.
- **Two UI options**: Fast web client with dark glassmorphic styling, or an analytical Streamlit dashboard.

## Project Structure

```
webchat/
├── backend/
│   ├── app/                 # FastAPI routes, agents, services, and models
│   ├── config/              # Database connection and setup
│   ├── .env.example         # Environment variable template
│   ├── Dockerfile           # Multi-stage Docker build with Astral UV
│   ├── main.py              # Application entry point (API and CLI)
│   ├── pyproject.toml       # UV project configuration and dependencies
│   └── uv.lock              # Pinned dependency lockfile
├── frontend/                # Web UI (HTML, CSS, JavaScript)
├── streamlit_app/           # Streamlit dashboard interface
├── data/                    # Local storage and cache files
├── docker-compose.yml       # Multi-container setup (Backend, Streamlit, Qdrant, Redis)
└── ARCHITECTURE.md          # Technical architecture and design details
```

## Getting Started

### Prerequisites

Install [`uv`](https://docs.astral.sh/uv/) for Python package management:

- **Windows:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/LaxmiNarayana31/webchat.git
cd webchat/backend
uv sync
```

### 2. Environment Configuration

Copy the template `.env.example` in `backend/`:

```bash
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Database Credentials (Aiven PostgreSQL)
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=5432
DB_NAME=defaultdb
SSL_MODE=require

# LLM Providers
GEMINI_API_KEY=
GROQ_API_KEY=

# Cache & Storage
REDIS_URL=rediss://default:YOUR_TOKEN@YOUR_HOST.upstash.io:6379
QDRANT_URL=
QDRANT_API_KEY=

# Security
URL_HASH_ALGORITHM=sha256
URL_HASH_SECRET=
```

### 3. Run Locally

From the `backend/` directory:

**Option A: Web Client & API**

```bash
uv run main.py --reload
```

- Web UI: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

**Option B: Streamlit Dashboard**

```bash
uv run main.py streamlit
```

- Dashboard: http://localhost:8501

### 4. Run with Docker

Run from the project root directory:

**Full stack with Docker Compose:**

```bash
docker compose up --build
```

**Backend only:**

```bash
docker build -t webchat-backend -f backend/Dockerfile .
docker run -d -p 8000:8000 --env-file backend/.env webchat-backend
```

## Architecture

For full details on the cyclic agent state graph, hybrid search algorithms, and database schemas, check out [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT
