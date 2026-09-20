<h1 align="center">WebChat</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/Vite-8.3+-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-4.0-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/uv-Astral-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="Astral UV" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-E30052?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/PostgreSQL-Aiven-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
</p>
<h3 align="center">Distributed Multi-Agent Conversational RAG & Search System</h3>
<p align="center">Production-ready conversational RAG platform featuring multi-agent orchestration, Corrective RAG (CRAG), Self-RAG reflection, hybrid dense/sparse retrieval with cross-encoder reranking, multi-provider LLM failover, and multimodal diagram extraction.</p>

---

## Overview

**WebChat** is an enterprise-grade conversational RAG platform engineered to parse, index, and query public web pages, technical articles, and documentation hubs. Built with a modular multi-agent supervisor architecture, WebChat decouples intent classification, query decomposition, hybrid retrieval, relevance grading, and synthesis into specialized agents with complete execution tracing and resilient failover.

---

## Features

### Multi-Agent Orchestration & State Management

- **Supervisor-Worker Pattern**: Protocol-driven structural typing governing specialized agents (Router, Planner, Researcher, Critic, and Synthesizer) with unified lifecycle management and execution tracing.
- **Intent Classification (Router Agent)**: Sub-50ms heuristic and semantic routing across document-grounded RAG, complex analytical reasoning, web search, and direct conversation.
- **Query Decomposition (Planner Agent)**: Breaks multi-part and comparative queries into parallel sub-queries with feedback-driven query reformulation upon retrieval critique.

### Corrective RAG (CRAG) & Self-RAG Reflection

- **Document Relevance Grading (Critic Agent)**: Evaluates retrieved passages against query intent to filter noise and prevent context poisoning prior to generation.
- **Hallucination & Groundedness Checks**: Validates factual alignment between generated responses and source excerpts, triggering corrective re-queries when grounding criteria are not met.

### High-Precision Hybrid Retrieval & Reranking

- **Dense & Sparse Search**: Combines semantic embeddings (Google `text-embedding-004` / Qdrant Cloud / FAISS) with keyword retrieval (BM25) for high recall across terminology and concepts.
- **Cross-Encoder Reranking**: Integrates FlashRank cross-encoder reranking to reorder top-k passages and optimize context precision for LLM generation.
- **Asynchronous Parallel Search**: Executes sub-query retrieval concurrently using non-blocking asynchronous routines.

### Multimodal Diagram & Structure Extraction

- **Visual Diagram Indexing**: Automatically extracts, processes, and embeds architectural diagrams, schematics, and figures from technical articles directly within markdown responses.
- **Multi-Stage Ingestion Pipeline**: Content extraction engine leveraging Trafilatura, Jina Reader, and BeautifulSoup4 with anti-bot and paywall bypass strategies.

### Multi-Provider Resilience & High Availability

- **10-Tier LLM Fallback Engine**: Transparent cascading across Google Gemini (`gemini-2.5-flash`, `gemini-3.6-flash`), Groq (`llama-3.3-70b-versatile`), OpenRouter, and Ollama with dynamic rate-limit backoff.
- **Real-Time Token-Bucket Rate Limiter**: Sliding-window limiter tracking requests per minute (RPM) and tokens per minute (TPM) per model to prevent upstream 429 errors.

### Defense-in-Depth Security & Rate Limiting

- **Anti-DDoS Burst Protection**: In-memory sliding-window token bucket intercepting high-frequency floods (>12 requests in 5 seconds per IP) at the middleware layer.
- **Dual-Layer Quota Enforcement**: Tracks usage against PostgreSQL across both Client ID and IP address (50 queries/day) to prevent browser-storage tampering while preserving frictionless guest access.
- **Scraper & Crawler Throttling**: Endpoint-specific limits protecting ingestion routes against resource exhaustion.

### High-Performance Caching & Data Persistence

- **Two-Tier Semantic Caching**: Upstash Redis semantic cache for instant cached response retrieval paired with in-memory vector cache.
- **PostgreSQL Session Storage**: Persistent storage for conversation turns, document metadata, and message histories with SHA-256 content deduplication.

### Dual Client Architectures

- **React SPA**: Production single-page application built with React 19, Vite, and Tailwind CSS, supporting Server-Sent Events (SSE) token streaming, KaTeX LaTeX math, interactive diagram viewing, and query persistence.
- **Streamlit Analytics Dashboard**: Standalone analytical interface with full parameter tuning, session isolation, and query debugging tools.

---

## Architecture

```
                                 [ Client Request ]
                                         │
                                         ▼
                            [ FastAPI API Gateway ]
                         (Anti-DDoS & IP Rate Limiting)
                                         │
                                         ▼
                               [ Router Agent ]
                       (Intent Classification & Routing)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
         [ Direct Conversation ]                    [ Supervisor Agent ]
                   │                         (Multi-Agent Workflow Orchestrator)
                   │                                           │
                   │                     ┌─────────────────────┼─────────────────────┐
                   │                     │                     │                     │
                   │             [ Planner Agent ]     [ Research Agent ]     [ Critic Agent ]
                   │           (Query Decomposition)  (Hybrid Search + Rerank)  (CRAG & Self-RAG)
                   │                     │                     │                     │
                   │                     └─────────────────────┼─────────────────────┘
                   │                                           │
                   │                                           ▼
                   │                                 [ Synthesis Agent ]
                   │                              (Context & Diagram Merge)
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                         [ Resilient Multi-Provider LLM ]
                       (Gemini / Groq / Ollama / DeepSeek)
                                         │
                                         ▼
                        [ SSE Real-Time Token Stream ]
```

---

## Project Structure

```
webchat/
├── backend/
│   ├── app/
│   │   ├── agents/          # Multi-agent system (Router, Planner, Research, Critic, Supervisor, Synthesis)
│   │   ├── api/             # FastAPI REST endpoints (chat, agents, users, frontend)
│   │   ├── cache/           # Semantic cache (Redis) & vector index cache
│   │   ├── clients/         # Resilient LLM client providers (Gemini, Groq, Ollama)
│   │   ├── core/            # Logging, error handling, rate limiting middleware, security
│   │   ├── dtos/            # Pydantic data transfer schemas and validation models
│   │   ├── helpers/         # Stream processing and URL hashing utilities
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── repositories/    # Database access layer (sessions, messages, URL caches)
│   │   └── services/        # Business logic (RAG pipeline, scraping, user quotas, memory)
│   ├── config/              # Database initialization and connection pool configuration
│   ├── main.py              # Application entry point and server lifecycle
│   ├── pyproject.toml       # Backend dependencies managed via Astral UV
│   └── uv.lock              # Pinned UV lockfile
├── frontend/                # React 19 + Vite + Tailwind CSS Single-Page Application
│   ├── src/
│   │   ├── App.jsx          # Primary chat interface, reasoning stepper, and diagram rendering
│   │   ├── App.css          # Glassmorphic component styles
│   │   ├── index.css        # Tailwind CSS directives
│   │   └── main.jsx         # React application entry point
│   ├── package.json         # Frontend package configuration
│   └── vite.config.js       # Vite build setup and proxy configuration
├── streamlit_app/           # Analytical Streamlit Dashboard
│   ├── app.py               # Streamlit application entry point
│   ├── components/          # Chat, sidebar, and layout components
│   └── styles.py            # Dashboard theme styling
├── data/                    # Persistent storage and FAISS vector indexes
└── ARCHITECTURE.md          # Architectural reference documentation
```

---

## Prerequisites

- **Python** `>= 3.11` (Python 3.11, 3.12, or 3.13)
- **Astral `uv`** package manager
- **Node.js** `>= 18.0.0` and **`npm`**

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/LaxmiNarayana31/webchat.git
cd webchat
```

### 2. Environment Configuration

Create a `.env` file in the `backend/` directory from the provided example template:

```bash
cp backend/.env.example backend/.env
```

Configure the necessary credentials in `backend/.env`:

```env
# Database Configuration (PostgreSQL / Aiven)
DB_USER=avnadmin
DB_PASSWORD=your_db_password
DB_HOST=your_db_host.aivencloud.com
DB_PORT=5432
DB_NAME=defaultdb
SSL_MODE=require

# Primary LLM & Embeddings (Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key

# Secondary LLM Provider (Groq)
GROQ_API_KEY=your_groq_api_key

# Semantic Cache (Upstash Redis)
REDIS_URL=rediss://default:YOUR_TOKEN@YOUR_HOST.upstash.io:6379

# Vector Database (Qdrant Cloud - Optional; falls back to FAISS if omitted)
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Cryptographic Salt
URL_HASH_SECRET=your_random_secret_string
```

---

### 3. Backend Setup & Execution

Navigate to `backend/` and install dependencies with `uv`:

```bash
cd backend
uv sync
```

Launch the FastAPI server:

```bash
uv run main.py
```

- **API Endpoint:** `http://localhost:8000`
- **Interactive Documentation:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

---

### 4. Frontend Setup & Execution

In a separate terminal, navigate to `frontend/`:

```bash
cd frontend
npm install
npm run dev
```

- **Frontend Application:** `http://localhost:5173`

---

### 5. Production Build (Unified Single-Port Hosting)

To build the React application for production hosting directly through FastAPI:

```bash
# 1. Build the production React assets
cd frontend
npm run build

# 2. Run the unified FastAPI server
cd ../backend
uv run main.py
```

FastAPI serves the compiled React application, static assets, documentation, and API routes from a single origin (`http://localhost:8000`).

---

### 6. Streamlit Dashboard

To launch the analytical Streamlit interface:

```bash
cd streamlit_app
uv run streamlit run app.py
```

- **Dashboard:** `http://localhost:8501`

---

## Environment Variables Reference

| Variable                  | Description                                                                                     | Required | Provider                                                 |
| :------------------------ | :---------------------------------------------------------------------------------------------- | :------: | :------------------------------------------------------- |
| `GEMINI_API_KEY`          | Primary LLM and dense embeddings (`text-embedding-004`, `gemini-2.5-flash`, `gemini-3.6-flash`) | **Yes**  | [Google AI Studio](https://aistudio.google.com/api-keys) |
| `GROQ_API_KEY`            | Secondary high-throughput LLM fallback (`llama-3.3-70b-versatile`)                              | **Yes**  | [Groq Console](https://console.groq.com/keys)            |
| `DB_USER` / `DB_PASSWORD` | PostgreSQL credentials for session, message, and URL cache persistence                          | **Yes**  | PostgreSQL / [Aiven](https://console.aiven.io/)          |
| `DB_HOST` / `DB_PORT`     | PostgreSQL host and port                                                                        | **Yes**  | PostgreSQL Provider                                      |
| `DB_NAME`                 | Database name (`defaultdb`)                                                                     | **Yes**  | PostgreSQL Provider                                      |
| `SSL_MODE`                | Database SSL connection mode (`require` or `prefer`)                                            | Optional | PostgreSQL Provider                                      |
| `REDIS_URL`               | Upstash Redis connection string for fast semantic caching                                       | Optional | [Upstash](https://console.upstash.com/)                  |
| `QDRANT_URL`              | Qdrant Cloud cluster endpoint for vector storage (FAISS fallback if omitted)                    | Optional | [Qdrant Cloud](https://cloud.qdrant.io/)                 |
| `QDRANT_API_KEY`          | Qdrant Cloud authentication API key                                                             | Optional | [Qdrant Cloud](https://cloud.qdrant.io/)                 |
| `URL_HASH_SECRET`         | Secret key used for SHA-256 URL content hashing                                                 | Optional | Application Configuration                                |

---

## API Reference

| Endpoint                  |  Method  | Description                                                           |
| :------------------------ | :------: | :-------------------------------------------------------------------- |
| `/health`                 |  `GET`   | Service liveness, version, and database connectivity status           |
| `/api/scrape`             |  `POST`  | Ingests URL, extracts text & diagrams, and builds vector index        |
| `/api/crawl`              |  `POST`  | Recursively crawls target domain up to specified depth                |
| `/api/chat`               |  `POST`  | Executes multi-agent conversational RAG query (JSON response)         |
| `/api/chat` (streaming)   |  `POST`  | SSE endpoint returning real-time token stream and reasoning telemetry |
| `/api/user/identify`      |  `POST`  | Links guest sessions to user identifiers and returns quota state      |
| `/api/user/quota`         |  `GET`   | Retrieves remaining daily quota for client/IP                         |
| `/api/user/sessions`      |  `GET`   | Retrieves conversation history list for active client ID              |
| `/api/user/sessions/{id}` | `DELETE` | Deletes a conversation session and associated turns                   |
| `/docs`                   |  `GET`   | Interactive OpenAPI Swagger UI                                        |

---

## License

This project is licensed under the **MIT License**.
