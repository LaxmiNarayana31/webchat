# WebChat

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

WebChat is a conversational RAG system that lets you ingest and chat with any public web page, technical documentation, or online article. It uses an asynchronous multi-agent supervisor pipeline with hybrid retrieval, cross-encoder reranking, multi-provider LLM failover, and automatic extraction of architectural diagrams.

## Overview

WebChat extracts article contents, processes embedded architectural diagrams, and builds a hybrid index across dense embeddings and sparse keywords. A supervisor agent routes queries, breaks complex questions into sub-queries, grades document relevance to prevent hallucinations, and streams answers with citations in real time.

## Features

### Multi-Agent Orchestration

- **Supervisor Pattern**: Coordinates task execution across specialized agents using typed Python protocols with full execution tracing.
- **Intent Routing**: Classifies queries into document RAG, analytical search, web lookup, or direct chat within 50 milliseconds.
- **Query Planner**: Decomposes complex multi-part questions into parallel sub-queries and refines queries based on retrieval feedback.

### Corrective RAG and Groundedness Verification

- **CRAG Relevance Grading**: Evaluates retrieved document passages against user intent, filtering out irrelevant noise before synthesis.
- **Self-RAG Reflection**: Checks factual alignment between generated drafts and source chunks to eliminate hallucinations.

### Hybrid Retrieval and Reranking

- **Dense and Sparse Search**: Queries dense vector embeddings in Qdrant Cloud or FAISS alongside BM25 sparse keyword indices.
- **Cross-Encoder Reranking**: Uses FlashRank to rerank candidate passages for high context precision.
- **Parallel Execution**: Gathers sub-query search results concurrently using asynchronous workers.

### Multimodal Diagram Extraction

- **Inline Architecture Diagrams**: Parses, indexes, and renders diagrams and technical figures directly inside chat answers.
- **Robust Scraping**: Ingests web pages using Trafilatura, Jina Reader, and BeautifulSoup with automatic paywall and anti-bot bypass.

### High Availability and Model Failover

- **Multi-Provider Fallback**: Automatically switches across Google Gemini, Groq LLaMA 3.3, OpenRouter, and Ollama when rate limits or provider downtime occur.
- **Rate Limiting**: Sliding-window token bucket manages requests per minute and tokens per minute per model.

### Edge Defense and Storage

- **Anti-DDoS Protection**: In-memory sliding-window limiter blocks burst floods exceeding 12 requests in 5 seconds per IP.
- **Dual-Layer Quota**: Tracks daily usage across both Client ID and IP address in PostgreSQL to prevent client tampering.
- **Two-Tier Caching**: Uses Upstash Redis for semantic caching alongside PostgreSQL for conversation turns and session storage.

### Client Interfaces

- **React Web App**: Single-page application built with React 19, Tailwind CSS, and Vite, featuring real-time SSE token streaming, LaTeX support, and interactive diagram modals.
- **Streamlit Dashboard**: Analytical interface offering model telemetry, direct vector inspection, and session management.

## Architecture

WebChat decouples routing, retrieval, relevance grading, and synthesis into dedicated agent roles.

For detailed architecture diagrams, sequence flows, and component catalogs, see [ARCHITECTURE.md](ARCHITECTURE.md).

### System Layers

1. **Client Layer**: React 19 SPA with SSE streaming and Streamlit analytical dashboard.
2. **Gateway and Edge Defense**: FastAPI gateway with sliding-window anti-DDoS burst protection and dual-layer quota tracking.
3. **Semantic Caching**: Upstash Redis semantic cache for sub-millisecond responses on repeated queries.
4. **Agent Orchestrator**: Supervisor agent coordinating Router, Planner, Research, Critic, and Synthesis agents.
5. **Retrieval Pipeline**: Dense embeddings in Qdrant/FAISS, sparse BM25 search, and FlashRank cross-encoder reranking.
6. **Resilient LLM Engine**: Multi-provider fallback chain across Gemini, Groq, OpenRouter, and Ollama.
7. **Persistence**: Aiven PostgreSQL for session state, message turns, and URL caches.

## Project Structure

```
webchat/
├── backend/
│   ├── app/
│   │   ├── agents/          # Router, Planner, Research, Critic, Supervisor, Synthesis agents
│   │   ├── api/             # FastAPI route handlers (chat, agents, users, frontend)
│   │   ├── cache/           # Redis semantic cache and vector store cache
│   │   ├── clients/         # Multi-provider LLM clients (Gemini, Groq, Ollama)
│   │   ├── core/            # Rate limiting, logging, security headers, and errors
│   │   ├── dtos/            # Pydantic schemas for requests and responses
│   │   ├── helpers/         # Stream formatters and cryptographic URL helpers
│   │   ├── models/          # SQLAlchemy database models
│   │   ├── repositories/    # Database repository access layer
│   │   └── services/        # RAG pipeline, scraper, user quota, and memory services
│   ├── config/              # Database initialization and connection pool
│   ├── main.py              # Application entry point
│   ├── pyproject.toml       # Backend dependencies managed with Astral UV
│   └── uv.lock              # Pinned lockfile
├── frontend/                # React 19 + Vite + Tailwind CSS application
│   ├── src/
│   │   ├── App.jsx          # Chat interface, reasoning stepper, and diagram rendering
│   │   ├── App.css          # Styling and layout rules
│   │   ├── index.css        # Tailwind CSS directives
│   │   └── main.jsx         # React application root
│   ├── package.json         # Frontend package configuration
│   └── vite.config.js       # Vite build setup
├── streamlit_app/           # Streamlit analytical dashboard
│   ├── app.py               # Streamlit application entry point
│   ├── components/          # Sidebar, chat, and layout components
│   └── styles.py            # Streamlit custom styles
├── main.py                  # Root entry point for cloud deployments
├── app.py                   # Root alias for Streamlit Cloud
├── requirements.txt         # Root dependency specification
└── ARCHITECTURE.md          # In-depth architectural documentation
```

## Prerequisites

- Python 3.11, 3.12, or 3.13
- Astral uv package manager
- Node.js 18 or newer with npm

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/LaxmiNarayana31/webchat.git
cd webchat
```

### 2. Environment Configuration

Copy the example environment file inside the `backend` directory:

```bash
cp backend/.env.example backend/.env
```

Fill in your credentials in `backend/.env`:

```env
DB_USER="avnadmin"
DB_PASSWORD="your_db_password"
DB_HOST="your_db_host.aivencloud.com"
DB_PORT="5432"
DB_NAME="defaultdb"
SSL_MODE="require"

GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"

REDIS_URL="rediss://default:YOUR_TOKEN@YOUR_HOST.upstash.io:6379"

QDRANT_URL="https://your-cluster-id.cloud.qdrant.io:6333"
QDRANT_API_KEY="your_qdrant_api_key"

URL_HASH_SECRET="your_random_secret_string"
```

### 3. Backend Setup

From the repository root, install backend dependencies and start the server:

```bash
cd backend
uv sync
uv run main.py
```

The API will be available at:

- API Endpoint: http://localhost:8000
- Swagger Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 4. Frontend Setup

In another terminal, start the development React server:

```bash
cd frontend
npm install
npm run dev
```

The React interface will be running at http://localhost:5173.

### 5. Single-Port Production Build

To build and serve the React application directly from FastAPI:

```bash
cd frontend
npm run build
cd ../backend
uv run main.py
```

FastAPI will serve the frontend bundle and all API endpoints together on http://localhost:8000.

### 6. Streamlit Dashboard

To run the Streamlit interface locally:

```bash
cd streamlit_app
uv run streamlit run app.py
```

The dashboard will open at http://localhost:8501.

## Environment Variables Reference

| Variable                  | Description                                 | Required | Provider                                                         |
| :------------------------ | :------------------------------------------ | :------: | :--------------------------------------------------------------- |
| `GEMINI_API_KEY`          | Primary LLM and embeddings generation       |   Yes    | [Google AI Studio](https://aistudio.google.com/api-keys)         |
| `GROQ_API_KEY`            | Fast fallback model provider                |   Yes    | [Groq Console](https://console.groq.com/keys)                    |
| `DB_USER` / `DB_PASSWORD` | PostgreSQL database credentials             |   Yes    | [Aiven](https://console.aiven.io/) or Local PostgreSQL           |
| `DB_HOST` / `DB_PORT`     | PostgreSQL host and port                    |   Yes    | PostgreSQL Provider                                              |
| `DB_NAME`                 | PostgreSQL database name                    |   Yes    | PostgreSQL Provider                                              |
| `SSL_MODE`                | Database connection SSL mode                | Optional | Defaults to require                                              |
| `REDIS_URL`               | Upstash Redis connection string for caching | Optional | [Upstash](https://console.upstash.com/)                          |
| `QDRANT_URL`              | Qdrant Cloud cluster endpoint               | Optional | [Qdrant Cloud](https://cloud.qdrant.io/) (FAISS used if omitted) |
| `QDRANT_API_KEY`          | Qdrant Cloud authentication key             | Optional | Qdrant Cloud                                                     |
| `URL_HASH_SECRET`         | Salt string for URL content hashing         | Optional | Application Configuration                                        |

## API Reference

| Endpoint                  | Method | Description                                     |
| :------------------------ | :----: | :---------------------------------------------- |
| `/health`                 |  GET   | Service status and database connection check    |
| `/api/scrape`             |  POST  | Scrapes URL, indexes text and diagrams          |
| `/api/crawl`              |  POST  | Recursively crawls target domain                |
| `/api/chat`               |  POST  | Executes RAG query (JSON or SSE streaming)      |
| `/api/user/identify`      |  POST  | Connects client ID to session history and quota |
| `/api/user/quota`         |  GET   | Returns remaining daily query quota             |
| `/api/user/sessions`      |  GET   | Lists chat sessions for client                  |
| `/api/user/sessions/{id}` | DELETE | Deletes a conversation session                  |
| `/docs`                   |  GET   | Interactive Swagger API documentation           |

## License

This project is licensed under the MIT License.
