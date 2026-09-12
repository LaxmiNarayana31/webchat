# WebChat AI — Autonomous Web Intelligence & Agentic RAG Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-orange?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Memori%20Labs-3.3+-4A154B?style=for-the-badge" alt="Memori Labs" />
  <img src="https://img.shields.io/badge/Google%20GenAI-Gemini%20Flash-4285F4?style=for-the-badge&logo=google" alt="Google GenAI" />
  <img src="https://img.shields.io/badge/Groq-Inference%20Engine-F55036?style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/PostgreSQL-Aiven-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Qdrant-Cloud%20Vector%20DB-DC2626?style=for-the-badge&logo=qdrant" alt="Qdrant Cloud" />
</p>

WebChat AI is an enterprise-grade conversational AI platform designed for autonomous web research and deep document intelligence. Users can provide any web URL (technical documentation, blogs, research papers, news articles, or gated publications) to extract content, build hierarchical vector indices, and engage in grounded multi-turn conversations backed by citations.

The platform orchestrates **LangGraph** agentic reasoning (Corrective RAG + Self-RAG), hybrid retrieval across **Qdrant Cloud** and sparse BM25, multi-provider LLM priority cascades (Google Gemini & Groq), persistent user semantic memory powered by **Memori Labs**, and managed database persistence on **Aiven PostgreSQL**.

For comprehensive technical specifications, class designs, and deep architectural analysis, refer to [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Architecture Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph Client_Layer ["1. Client & Presentation Layer"]
        UI_SPA["Web SPA (Vanilla HTML5 / Modern CSS / JavaScript ES6+)<br/>• Real-time SSE Stream Consumer<br/>• Interactive Citation Cards<br/>• Responsive Glassmorphic Dark Theme<br/>• Model Switcher & Ingestion Bar"]
        UI_Streamlit["Streamlit Dashboard (Python / Custom CSS)<br/>• Model Telemetry & Quota Gauges<br/>• Document & Web Page Ingestion<br/>• Interactive Chat Stream & Explorer"]
    end

    subgraph API_Gateway ["2. Backend Gateway Layer (FastAPI / Starlette)"]
        ASGI["Uvicorn ASGI Server (:8000)"]
        MW_CORS["CORSMiddleware (Configurable Origins)"]
        MW_Sec["SecurityHeadersMiddleware<br/>(nosniff, DENY, XSS, strict-origin)"]
        
        subgraph Services ["Application Routing & Services"]
            H_Front["Frontend Service<br/>• Single-Page Web Application<br/>• Static Asset Delivery (/css, /js)"]
            H_Chat["Chat & Ingestion Service<br/>• User URL Scraping & Paywall Bypass<br/>• Domain Crawling & PDF Ingestion<br/>• Real-Time Token Streaming Engine"]
            H_User["User & Session Service<br/>• Email Identification<br/>• Daily Quota Management<br/>• Conversation Session History"]
        end
    end

    subgraph Core_Orchestration ["3. Application & Agentic Orchestration"]
        SVC_Chat["ChatService<br/>• Quota Verification & Consumption<br/>• Session & Message Persistence<br/>• Semantic Cache Interception<br/>• Stream Event Formatting"]
        AGENT_Chat["WebChatAgent<br/>• Async / Sync Streaming Bridge<br/>• LangGraph Orchestration<br/>• Citation Formatting & Chunk Mapping"]
        SVC_AgenticRAG["AgenticRAGService (LangGraph)<br/>• Intent Routing & Sub-query Decomposition<br/>• Corrective RAG (CRAG Relevance Grading)<br/>• Self-RAG Groundedness Evaluation"]
        SVC_User["UserService<br/>• Anonymous Guest Quotas (5 Queries)<br/>• Registered User Quotas (50 Queries/Day)<br/>• Automatic UTC Midnight Reset"]
    end

    subgraph Intelligence_RAG ["4. RAG, Retrieval & Memory Services"]
        SVC_RAG["RAGService<br/>• RecursiveCharacterTextSplitter (800 / 150)<br/>• GeminiEmbeddings (3072-dim Vectors)<br/>• Hybrid Search (Dense Vectors + BM25 Lexical + RRF)"]
        SVC_Rerank["RerankService<br/>• FlashRank (ms-marco-TinyBERT-L-2-v2 Cross-Encoder)"]
        SVC_Expand["QueryExpansionService<br/>• LLM Sub-query Decomposition"]
        SVC_Scraper["ScraperService<br/>• Trafilatura & BeautifulSoup4<br/>• Medium Apollo State Deserialization<br/>• Jina Reader Headless Engine<br/>• Wayback Machine Snapshot Retrieval<br/>• BFS Parallel Crawler (ThreadPoolExecutor)"]
        SVC_Memory["MemoryService (Memori Labs)<br/>• Native LLM Invocation Interception<br/>• User Entity Attribution<br/>• Semantic Memory Recall<br/>• Background Knowledge Graph Ingestion"]
        SVC_Qdrant["QdrantService<br/>• Qdrant Cloud Vector Database"]
    end

    subgraph LLM_Cascade ["5. Resilient Multi-Provider LLM Engine"]
        SVC_LLM["LLMService / ResilientLLMClient"]
        Limiter["Rate Limiting & Cooldown Protection<br/>• TokenBucket Rate Limiters<br/>• SlidingWindowTracker (RPM)<br/>• CooldownTracker (Exponential Backoff + Jitter)"]
        
        subgraph Providers ["Fallback Model Priority Chain"]
            M_Gemini["Google Gemini (google-genai SDK)<br/>• gemini-3.6-flash (Primary)<br/>• gemini-3.5-flash<br/>• gemini-3.5-flash-lite<br/>• gemini-3.1-flash-lite<br/>• gemini-2.5-flash"]
            M_Groq["Groq Cloud (groq SDK)<br/>• openai/gpt-oss-120b<br/>• groq/compound<br/>• openai/gpt-oss-20b<br/>• groq/compound-mini"]
        end
    end

    subgraph Storage_Layer ["6. Cloud Storage, Databases & Caches"]
        subgraph Relational_DB ["Aiven PostgreSQL Database"]
            T_Users[("webchat_users")]
            T_Sessions[("webchat_sessions")]
            T_Messages[("webchat_messages")]
            T_Guest[("webchat_guest_usage")]
            T_UrlCache[("webchat_url_cache")]
        end

        subgraph Memori_Store ["Memori Labs Schema (PostgreSQL BYODB)"]
            T_MemEntity[("memori_entity")]
            T_MemSession[("memori_session")]
            T_MemConv[("memori_conversation")]
            T_MemMsg[("memori_conversation_message")]
            T_MemFacts[("memori_entity_fact")]
            T_MemGraph[("memori_knowledge_graph")]
        end

        subgraph Vector_Stores ["Qdrant Cloud & Caches"]
            IDX_Qdrant["Qdrant Cloud (webchat_chunks collection)"]
            CACHE_DiskVector["Disk Vector Cache (data/vector_storage/)"]
            CACHE_Semantic["Disk Semantic Answer Cache (data/semantic_cache/)"]
        end
    end

    subgraph External_APIs ["7. External Cloud Services & APIs"]
        EXT_Google["Google Gemini AI Platform"]
        EXT_Groq["Groq Cloud Inference Engine"]
        EXT_DDG_Wiki["DuckDuckGo & Wikipedia APIs"]
        EXT_Jina["Jina Reader API (r.jina.ai)"]
        EXT_Wayback["Internet Archive Wayback Machine"]
    end

    %% Wiring
    UI_SPA --> ASGI
    UI_Streamlit --> ASGI
    UI_Streamlit -.-> AGENT_Chat

    ASGI --> MW_CORS --> MW_Sec
    MW_Sec --> H_Front
    MW_Sec --> H_Chat
    MW_Sec --> H_User

    H_Chat --> SVC_Chat
    H_User --> SVC_User
    SVC_Chat --> SVC_User
    SVC_Chat --> AGENT_Chat
    AGENT_Chat --> SVC_AgenticRAG

    SVC_AgenticRAG --> SVC_RAG
    SVC_AgenticRAG --> SVC_Rerank
    SVC_AgenticRAG --> SVC_Expand
    SVC_AgenticRAG --> SVC_Memory
    SVC_AgenticRAG --> SVC_LLM
    SVC_AgenticRAG -.-> EXT_DDG_Wiki

    SVC_RAG --> SVC_Qdrant
    SVC_Qdrant --> IDX_Qdrant

    SVC_Chat --> CACHE_Semantic
    AGENT_Chat --> CACHE_DiskVector
    SVC_Chat --> Relational_DB

    SVC_LLM --> Limiter
    Limiter --> M_Gemini
    Limiter --> M_Groq
    M_Gemini --> EXT_Google
    M_Groq --> EXT_Groq

    SVC_Scraper --> EXT_Jina
    SVC_Scraper --> EXT_Wayback

    SVC_Memory --> Memori_Store
```

---

## User URL & End-to-End Data Flow

This diagram illustrates how WebChat AI handles user research queries against any targeted web URL, orchestrating ingestion, hybrid retrieval, agentic reasoning, and token streaming.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Researcher
    participant UI as Web SPA / Streamlit UI
    participant Gateway as FastAPI Gateway
    participant ChatSvc as ChatService
    participant UserSvc as UserService
    participant CacheSvc as SemanticCacheService
    participant Agent as WebChatAgent
    participant LangGraph as AgenticRAGService (LangGraph)
    participant Scraper as ScraperService
    participant VectorDB as Qdrant Cloud
    participant Memori as MemoryService (Memori Labs)
    participant LLMEngine as ResilientLLMClient
    participant DB as Aiven PostgreSQL

    User->>UI: Enter Target Web URL & Research Question
    UI->>Gateway: Submit Request (User URL + Query)
    Gateway->>ChatSvc: Process Chat Query
    
    %% Quota Verification
    ChatSvc->>UserSvc: Verify Quota (Email / Client ID)
    UserSvc->>DB: Check & Increment Usage Record
    DB-->>UserSvc: Quota Approved
    UserSvc-->>ChatSvc: Quota Allowed

    %% Ingestion / Scraping if needed
    opt URL Not Yet In Cache
        ChatSvc->>Scraper: Scrape & Bypass Paywalls (User URL)
        Scraper-->>ChatSvc: Clean Web Markdown Content
        ChatSvc->>VectorDB: Insert Chunks & Embeddings (User URL)
    end

    %% User Attribution & User Message
    ChatSvc->>Memori: Set User Attribution
    ChatSvc->>DB: Save User Query

    %% Semantic Cache
    ChatSvc->>CacheSvc: Check Semantic Answer Cache
    alt Exact Cache Hit
        CacheSvc-->>ChatSvc: Return Cached Answer & Citations
        ChatSvc-->>UI: Stream Cached Response & Citations
    else Cache Miss
        ChatSvc->>Agent: Run Agentic RAG Pipeline
        Agent->>LangGraph: Execute State Machine Workflow
        
        %% RAG Pipeline
        LangGraph->>LangGraph: Classify Intent (Direct vs Web vs Document)
        LangGraph->>LangGraph: Decompose Complex Questions
        LangGraph->>VectorDB: Retrieve Context Chunks (User URL)
        VectorDB-->>LangGraph: Relevant Document Chunks
        LangGraph->>LangGraph: Grade Relevance (CRAG)
        
        opt Insufficient Context
            LangGraph->>LangGraph: Rewrite Query & Query DuckDuckGo / Wikipedia
        end

        %% Context Synthesis & Memory Recall
        LangGraph->>Memori: Recall Past Facts & User Preferences
        Memori-->>LangGraph: Relevant Memory Context
        LangGraph->>LangGraph: Synthesize Grounded Context & System Prompt
        LangGraph-->>Agent: Reasoning Steps & Citations Ready
        Agent-->>UI: Display Reasoning Steps & Citations

        %% Streaming Generation
        Agent->>LLMEngine: Stream Generation with Resilient Fallback
        LLMEngine-->>UI: Real-Time Stream Tokens
        LLMEngine->>Memori: Capture Turn & Extract Facts Automatically
        Agent-->>ChatSvc: Stream Completed

        %% Persistence & Caching
        ChatSvc->>CacheSvc: Cache Answer & Citations
        ChatSvc->>DB: Save Assistant Answer & Citations
        ChatSvc-->>UI: Complete Stream
    end
    UI-->>User: Display Formatted Answer & Source Citations
```

---

## Agentic RAG Pipeline (LangGraph Workflow)

```mermaid
stateDiagram-v2
    [*] --> START
    START --> route_node: User Query & Target URL Input

    state route_node <<choice>>
    note right of route_node
        Classifies query intent:
        • Direct conversational
        • Web search trigger keywords
        • Ingested User URL Document RAG
    end note

    route_node --> DIRECT_CHAT: Simple greeting / Conversational
    route_node --> WEB_SEARCH: Explicit web search request
    route_node --> DOCUMENT_RAG: User URL analysis question

    state DIRECT_CHAT {
        synthesize_direct: Synthesize direct conversational prompt
    }

    state WEB_SEARCH {
        web_search_direct: DuckDuckGo & Wikipedia Web Search
    }

    state DOCUMENT_RAG {
        decompose: decompose_and_expand_node<br/>(Split complex comparative queries)
        retrieve: retrieve_node<br/>(Qdrant Cloud Dense + BM25 Sparse + FlashRank Rerank)
        grade_docs: grade_documents_node<br/>(CRAG Cross-Encoder Relevance Grading)
        
        state grade_docs <<choice>>
        grade_docs --> Pass: Confidence >= 0.5
        grade_docs --> Fallback: Ambiguous / Low Context
        
        Fallback --> transform_query: transform_query_node (LLM query rewrite)
        transform_query --> retrieve: Retry retrieval with rewritten query
        Fallback --> web_fallback: web_search_node (DuckDuckGo fallback)
    }

    DIRECT_CHAT --> synthesize_context
    WEB_SEARCH --> synthesize_context
    Pass --> synthesize_context
    web_fallback --> synthesize_context

    state synthesize_context {
        inject_memory: Recall & inject Memori user preferences
        build_prompt: Construct grounded prompt & system instructions
    }

    synthesize_context --> generate_answer: LLM Multi-Provider Stream

    state grade_hallucination {
        self_rag: grade_hallucination_and_faithfulness<br/>(Check groundedness in context)
    }

    generate_answer --> grade_hallucination

    state grade_hallucination <<choice>>
    grade_hallucination --> Grounded: Answer supported by context
    grade_hallucination --> Regenerate: Hallucination detected (attempt 1)
    
    Regenerate --> generate_answer: Re-prompt with strict grounding
    Grounded --> END: Complete Response Output
    END --> [*]
```

---

## Key Capabilities

### Chat with Any User URL
- **Instant Web Intelligence**: Users enter any public URL to scrape, clean, chunk, embed, and chat with the page contents in real time.
- **Paywall & Gating Bypass**:
  - Direct scraping with realistic browser headers via `Trafilatura` and `BeautifulSoup4`.
  - Deserialization of Medium Apollo State (`window.__APOLLO_STATE__`) to retrieve complete subscriber-only article text.
  - Headless markdown extraction via Jina Reader (`r.jina.ai`) for client-side JavaScript rendering.
  - Automatic historical snapshot extraction from the Internet Archive Wayback Machine.
- **Document Ingestion**: Upload PDF or text files for instant chunking, embedding, and cross-document querying.
- **Deep Domain Crawling**: Breadth-First Search (BFS) crawler using `concurrent.futures.ThreadPoolExecutor` to crawl and index entire documentation domains.

### LangGraph Agentic RAG & Hybrid Retrieval
- **Cyclic StateGraph**: Orchestrates routing, query decomposition, retrieval, grading, query rewriting, and hallucination evaluation.
- **Hybrid Search with RRF**: Combines 3072-dimensional dense embeddings (`GeminiEmbeddings`) and sparse lexical search (`rank-bm25`) using Reciprocal Rank Fusion ($k=60$).
- **Cross-Encoder Reranking**: Low-latency local cross-encoder scoring via FlashRank (`ms-marco-TinyBERT-L-2-v2`).
- **Corrective RAG (CRAG)**: Grades retrieved context relevance; rewrites queries and searches DuckDuckGo and Wikipedia when context is missing.
- **Self-RAG Groundedness**: Verifies generated answers against source text to prevent hallucinations.

### 10-Tier Resilient LLM Cascade
- Automatic priority fallback between Google Gemini and Groq if rate limits or provider issues occur:
  - `gemini-3.6-flash` (Primary)
  - `openai/gpt-oss-120b` (Groq Fallback)
  - `gemini-3.5-flash`
  - `groq/compound`
  - `gemini-3.5-flash-lite`
  - `openai/gpt-oss-20b`
  - `gemini-3.1-flash-lite`
  - `groq/compound-mini`
  - `gemini-2.5-flash` / `gemini-2.5-flash-lite`
- **Circuit Breakers**: TokenBucket rate limiting, sliding-window RPM monitoring, and exponential backoff with jitter.

### Persistent Semantic Memory (Memori Labs)
- **Automatic Interception**: Connects with the LLM client to record conversations and identify facts automatically.
- **User Attribution**: Links memory graphs and preferences to the user identity.
- **Cloud Schema**: Structured relational entities, sessions, messages, and knowledge graph facts stored in Aiven PostgreSQL.

### Tiered Quotas & Security
- **Guest Tier**: 5 queries tracked by client device UUID.
- **Registered User Tier**: 50 queries per day tracked by email, automatically reset at `00:00:00 UTC`.
- **Hardened HTTP Headers**: Injected via `SecurityHeadersMiddleware` (`nosniff`, `DENY`, `X-XSS-Protection`, strict referrer policy).

---

## Repository Layout

```
webchat/
├── backend/
│   ├── app/
│   │   ├── agents/          # Autonomous agents & stream bridges (chat_agent.py)
│   │   ├── api/             # HTTP routing controllers (chat, user, frontend)
│   │   ├── cache/           # Semantic cache & vector cache stores
│   │   ├── clients/         # Provider clients (gemini_client.py, groq_client.py, llm_client.py)
│   │   ├── core/            # Config, DB connection, security, rate limiting, logging
│   │   ├── dtos/            # Pydantic v2 validation schemas
│   │   ├── helpers/         # URL, text, and stream formatting utilities
│   │   ├── models/          # SQLAlchemy entities (users, sessions, messages, caches)
│   │   ├── repositories/    # Database repository layer
│   │   └── services/        # LangGraph, RAG, reranker, scraper, memory, user, and chat services
│   ├── config/              # Central database connection factory (Aiven PostgreSQL)
│   ├── .env.example         # Template for required environment variables
│   ├── .python-version      # Target Python runtime version (3.13)
│   ├── main.py              # Unified FastAPI application factory, routes & CLI launcher
│   ├── pyproject.toml       # Backend Astral UV manifest & dependency definitions
│   ├── pyrightconfig.json   # Backend-scoped Pyright configuration
│   ├── requirements.txt     # Exported dependency requirements
│   └── uv.lock              # Reproducible UV dependency lockfile
│
├── frontend/                # Single-Page Web Client (HTML5, Vanilla CSS, JS ES6+)
│   ├── index.html           # Modern glassmorphic interface layout
│   ├── index.css            # Dark theme styles, responsive drawer, animations
│   └── app.js               # Real-time SSE streaming reader, citation drawer
│
├── streamlit_app/           # Analytical Dashboard & Visualizer
│   ├── app.py               # Streamlit application entry point
│   ├── styles.py            # Custom CSS theming
│   ├── requirements.txt     # Streamlit deployment requirements
│   └── components/          # Modular UI components (header, sidebar, chat)
│
├── data/                    # Persistent disk storage (semantic cache, vector indices)
├── .gitignore               # Git exclude rules for secrets, caches, and venvs
├── ARCHITECTURE.md          # Comprehensive technical design document
├── pyrightconfig.json       # Monorepo Pyright configuration
└── README.md                # System documentation
```

---

## Process & Runtime Architecture (`backend/main.py`)

The application provides a unified execution entrypoint in [backend/main.py](file:///d:/Projects/Personal-Projects/webchat/backend/main.py) that detects execution arguments and starts the chosen interface, as well as exposing `app` for direct ASGI deployment.

```mermaid
graph TB
    subgraph CLI_Entry ["Process Launcher (backend/main.py)"]
        CLI["python backend/main.py [service]"]
        Detect{"Detect Mode & Context"}
        CLI --> Detect
    end

    subgraph Service_Execution ["Runtime Processes"]
        Detect -->|service == 'backend' or 'api' (default)| Backend_Proc["FastAPI Application (Uvicorn ASGI)<br/>uvicorn.run('backend.main:app', host, port)<br/>Default: 0.0.0.0:8000"]
        Detect -->|service == 'streamlit'| Streamlit_Proc["Streamlit Analytical Dashboard<br/>subprocess.call([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app/app.py'])<br/>Default: localhost:8501"]
    end

    subgraph Backend_Subsystems ["FastAPI Application Subsystems"]
        Lifespan["Application Lifespan<br/>• Database connection & table setup<br/>• Memori Labs BYODB storage build<br/>• Semantic & vector cache pre-warming"]
        Static_Mounts["Static Asset Mounts<br/>• /css -> frontend/css<br/>• /js -> frontend/js<br/>• /static -> frontend/"]
        App_Services["Application Core Services<br/>• Chat & Ingestion Engine<br/>• User & Quota Management<br/>• Single-Page Web Application UI"]
        
        Backend_Proc --> Lifespan
        Lifespan --> Static_Mounts
        Lifespan --> App_Services
    end

    subgraph Cloud_Storage ["Cloud Storage & Databases"]
        Cloud_DB[("Aiven PostgreSQL Database<br/>Configured via DB_USER, DB_PASSWORD, DB_HOST, DB_NAME<br/>• Application relational tables<br/>• Memori Labs BYODB tables")]
        Cloud_Qdrant[("Qdrant Cloud Vector Database<br/>Configured via QDRANT_URL, QDRANT_API_KEY")]
        Local_Cache["Local Cache Filesystem (data/)<br/>• data/semantic_cache/*.json<br/>• data/vector_storage/<hash>/"]
    end

    subgraph External_APIs ["External Cloud Services"]
        API_Gemini["Google Gemini AI API (google-genai SDK)"]
        API_Groq["Groq Cloud API (groq SDK)"]
        API_DDG_Wiki["DuckDuckGo Instant Answer & Wikipedia APIs"]
        API_Jina["Jina Reader API (r.jina.ai)"]
        API_Wayback["Internet Archive Wayback Machine"]
    end

    App_Services --> Cloud_DB
    App_Services --> Cloud_Qdrant
    App_Services --> Local_Cache
    App_Services --> External_APIs
    Streamlit_Proc -.->|Agent Bridge| App_Services
```

---

## Installation & Setup (New User Quickstart)

### 1. Prerequisites
- **Python**: Version `3.11`, `3.12`, or `3.13` installed.
- **Git**: Installed.
- **Package Manager**: [Astral UV](https://docs.astral.sh/uv/) (Recommended for ultra-fast setup) or standard `pip`.

### 2. Clone the Repository
```bash
git clone https://github.com/LaxmiNarayana31/webchat.git
cd webchat
```

### 3. Create Virtual Environment & Install Dependencies

#### Using Astral UV (Recommended)
```bash
cd backend
uv sync
cd ..
```

#### Using Standard Python `venv` & `pip`
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables (`.env`)

Copy the template file to `backend/.env` (or root `.env`):

**On Windows (PowerShell):**
```powershell
Copy-Item backend/.env.example backend/.env
```

**On Linux / macOS:**
```bash
cp backend/.env.example backend/.env
```

Generate your secure `URL_HASH_SECRET` with Python:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Edit `backend/.env` and paste your credentials:

```env
# https://console.aiven.io/ (Optional - defaults to local SQLite)
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=5432
DB_NAME=defaultdb
SSL_MODE=require

# https://aistudio.google.com/api-keys
GEMINI_API_KEY=your_gemini_key

# https://console.groq.com/keys
GROQ_API_KEY=your_groq_key

# https://cloud.qdrant.io/ (Optional - defaults to local FAISS)
QDRANT_URL=
QDRANT_API_KEY=

# Cryptographic URL Cache Hashing
URL_HASH_ALGORITHM=sha256
URL_HASH_SECRET=your_32_byte_generated_hex_secret
```

### 5. Pre-flight Verification & Database Initialization

Run the pre-flight verification command to test your environment and initialize database tables:
```bash
# Verify backend imports and initialize tables:
python -c "from backend.config.database import init_db; init_db(); print('Environment and database ready!')"
```

---

## Running the Application

### Option A: Modern Single-Page Web Application (FastAPI Backend + Built-in Web UI)

The FastAPI server automatically serves the modern glassmorphic Web SPA at the root URL (`http://localhost:8000`).

```bash
# Using UV
uv run --project backend python backend/main.py

# Or with active virtual environment:
python backend/main.py --host 0.0.0.0 --port 8000 --reload
```

- **Web App**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

### Option B: Streamlit Analytical Dashboard

```bash
# Using UV
uv run --project backend python backend/main.py streamlit

# Or with active virtual environment:
python backend/main.py streamlit
```

- **Dashboard**: Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Quality & Architectural Standards

The codebase adheres strictly to enterprise Python best practices:
- **Zero `__init__.py` Files**: Modern namespace packages across all internal subdirectories.
- **Deterministic Imports**: Strictly top-level module imports; zero inline imports or guarded try-except import blocks.
- **Robust Exception Handling**: Start-to-end `try...except` exception boundaries in all functions with contextual logging.
- **Zero Deprecated Constructs**: Built on `google-genai` official SDK, `Pydantic v2`, and `FastAPI lifespan`.

---

## License

This project is licensed under the MIT License.
