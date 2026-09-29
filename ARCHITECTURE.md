<p align="center">
  <img src="assets/logo.svg" width="92" height="92" alt="WebChat Logo" />
</p>

<h1 align="center">WebChat: System Architecture and Technical Design</h1>

This document provides a comprehensive architectural analysis of the **WebChat** platform based on the production codebase implementation.

---

## 1. High-Level System Architecture

WebChat provides a dual-interface conversational AI platform supporting both a **React 19 single-page application** and a **Streamlit analytical dashboard**. The backend is built with **FastAPI** and orchestrates an asynchronous multi-agent supervisor pipeline with hybrid retrieval (dense vectors, sparse BM25, and FlashRank cross-encoder reranking), multi-provider LLM fallback cascades (Google Gemini, Groq, and Ollama), two-tier semantic caching, a 7-tier resilient web scraper, and relational persistence via PostgreSQL.

```mermaid
graph TB
    subgraph Client_Layer ["1. Client & Presentation"]
        UI_SPA["React 19 SPA (SSE Streaming + KaTeX + Titanium Theme)"]
        UI_Streamlit["Streamlit Analytics Dashboard"]
    end

    subgraph API_Gateway ["2. API Gateway & Defense-in-Depth"]
        FastAPI["FastAPI / Uvicorn Server"]
        DDoS["Anti-DDoS Sliding Window (>12 req/5s per IP)"]
        Quota["Dual-Layer Quota (IP + Client ID / 50 daily)"]
    end

    subgraph Core_Orchestration ["3. Multi-Agent Supervisor Orchestration"]
        Supervisor["Supervisor Agent (State & Trace Coordinator)"]
        Router["Router Agent (Sub-50ms Intent Classifier)"]
        Planner["Planner Agent (Query Decomposition & Rewriter)"]
        Researcher["Research Agent (Parallel Hybrid Retrieval)"]
        Critic["Critic Agent (CRAG Relevance & Self-RAG Reflection)"]
        Synthesizer["Synthesis Agent (Context & Diagram Merge)"]
    end

    subgraph Intelligence_RAG ["4. Intelligence & Retrieval"]
        SVC_RAG["Hybrid RAG (Qdrant Cloud Dense + BM25 Sparse)"]
        SVC_Rerank["FlashRank Cross-Encoder Reranker"]
        SVC_Scraper["7-Tier Scraper (Substack API + Freedium + Jina + Trafilatura)"]
        SVC_Memory["Semantic Memory (Memori Labs)"]
    end

    subgraph LLM_Cascade ["5. Resilient LLM Engine (10-Tier Failover)"]
        SVC_LLM["Resilient LLM Client (Token-Bucket Limiter)"]
        M_Gemini["Google Gemini 2.0 / 2.5 Flash (Primary)"]
        M_Groq["Groq LLaMA 3.3 70B (Fallback)"]
        M_Ollama["Ollama / DeepSeek / OpenRouter (Cascades)"]
    end

    subgraph Storage_Layer ["6. Storage & Caches"]
        DB_Relational[("PostgreSQL Database (Aiven Cloud)")]
        DB_Vector[("Qdrant Cloud / FAISS Vector Store")]
        CACHE_Redis[("Upstash Cloud Redis (Two-Tier Semantic Cache)")]
    end

    UI_SPA --> FastAPI
    UI_Streamlit --> FastAPI
    FastAPI --> DDoS
    DDoS --> Quota
    Quota --> CACHE_Redis

    CACHE_Redis -->|Cache Miss| Supervisor
    Supervisor --> Router
    Supervisor --> Planner
    Supervisor --> Researcher
    Supervisor --> Critic
    Supervisor --> Synthesizer

    Researcher --> SVC_RAG
    Researcher --> SVC_Rerank
    Supervisor --> SVC_Scraper
    Supervisor --> SVC_Memory

    Synthesizer --> SVC_LLM
    SVC_LLM --> M_Gemini
    SVC_LLM --> M_Groq
    SVC_LLM --> M_Ollama

    CACHE_Redis -->|Tier 1 Exact & Tier 2 Vector| UI_SPA
    Supervisor --> DB_Relational
    SVC_RAG --> DB_Vector
    SVC_Memory --> DB_Relational
```

---

## 2. End-to-End Data and Request Flow

The sequence diagram below maps the complete lifecycle of a chat request from the client interface down to security middleware, quota checks, multi-agent orchestration, hybrid retrieval, streaming generation, and persistence.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (React 19 / Streamlit)
    participant Shield as RateLimiterMiddleware (Anti-DDoS)
    participant API as FastAPI Gateway (/api/chat)
    participant ChatSvc as ChatService
    participant UserSvc as UserService (Dual Quota)
    participant CacheSvc as SemanticCache (Upstash Redis)
    participant Supervisor as SupervisorAgent (Orchestrator)
    participant Router as RouterAgent
    participant Planner as PlannerAgent
    participant Research as ResearchAgent (Hybrid + FlashRank)
    participant Critic as CriticAgent (CRAG & Self-RAG)
    participant Synth as SynthesisAgent
    participant LLMEngine as ResilientLLMClient (10-Tier Failover)
    participant DB as PostgreSQL (Aiven Cloud)

    User->>Shield: POST /api/chat {query, stream: true, client_id, ...}
    Shield->>Shield: Validate sliding window (>12 req/5s per IP)
    alt Flood Attack Detected (Burst > 12 req/5s)
        Shield-->>User: HTTP 429 Too Many Requests (Retry-After: 5s)
    else Request Allowed
        Shield->>API: Forward request with client_ip
        API->>ChatSvc: handle_chat_stream_async(req, client_ip)

        ChatSvc->>UserSvc: check_and_consume_quota(client_id, ip_address)
        UserSvc->>DB: Check & increment quota in webchat_guest_usage
        DB-->>UserSvc: Quota Approved (50 daily limit per IP + CID)
        UserSvc-->>ChatSvc: allowed=True, quota_info

        ChatSvc->>CacheSvc: get_cached_answer(query, context_hash)
        alt Tier 1 / Tier 2 Cache Hit (Similarity >= 0.92)
            CacheSvc-->>ChatSvc: cached_answer, citations
            CacheSvc-->>User: SSE Event: instant response + citations
        else Cache Miss
            ChatSvc->>Supervisor: orchestrate_stream_async(query, vector_store, ...)

            Supervisor->>Router: classify_intent_async(query)
            Router-->>Supervisor: Intent Decision (DOCUMENT_RAG)

            Supervisor->>Planner: decompose_query_async(query)
            Planner-->>Supervisor: sub_queries [q1, q2]

            Supervisor->>Research: retrieve_documents_async(sub_queries)
            Research->>Research: Parallel Qdrant Dense + BM25 Sparse + FlashRank Rerank
            Research-->>Supervisor: Top-K Document Passages

            Supervisor->>Critic: grade_documents_async(passages) [CRAG]
            alt Irrelevant Passages (Confidence < 0.4)
                Critic-->>Supervisor: is_relevant=False, reasoning
                Supervisor->>Planner: rewrite_query_async(query, critique)
                Planner-->>Supervisor: rewritten_query
                Supervisor->>Research: re-retrieve with rewritten query
            else Passages Relevant
                Critic-->>Supervisor: is_relevant=True, filtered_passages
            end

            Supervisor->>Synth: synthesize_context_async(passages, metadata)
            Synth->>Synth: Extract architecture diagrams & format citations
            Synth-->>Supervisor: assembled_prompt, system_instruction, citations

            Supervisor-->>ChatSvc: yield {"type": "ready", prompt, citations}
            ChatSvc-->>User: SSE Event: reasoning steps & citations

            ChatSvc->>LLMEngine: generate_stream_async(prompt, system_instruction)
            loop Token Generation
                LLMEngine-->>ChatSvc: token chunk
                ChatSvc-->>User: SSE Event: text chunk
            end

            ChatSvc->>DB: append_message(role='user', content=query)
            ChatSvc->>DB: append_message(role='assistant', content=answer, citations)
            ChatSvc->>CacheSvc: set_cached_answer(query, answer, citations)
            ChatSvc-->>User: SSE Event: [DONE]
        end
    end
```

---

## 3. Detailed AI and Agentic RAG Pipeline (Multi-Agent Supervisor Workflow)

The intelligence layer operates as an asynchronous multi-agent supervisor architecture implementing Corrective RAG (CRAG), Self-RAG reflection, and Parallel Hybrid Retrieval.

```mermaid
stateDiagram-v2
    [*] --> START
    START --> Router_Agent: User Query Input

    state Router_Agent {
        intent_check: Sub-50ms Heuristic & Semantic Intent Classification
    }

    Router_Agent --> DIRECT_CHAT: Simple greeting / Conversational
    Router_Agent --> WEB_SEARCH: Explicit real-time / web inquiry
    Router_Agent --> DOCUMENT_RAG: Document analysis question

    state DIRECT_CHAT {
        direct_synth: Direct Prompt Synthesis (Bypasses Retrieval)
    }

    state WEB_SEARCH {
        web_retrieval: DuckDuckGo Web Search & Fallback Synthesis
    }

    state DOCUMENT_RAG {
        Planner_Decompose: PlannerAgent (Sub-query decomposition)
        Research_Hybrid: ResearchAgent (Parallel Qdrant Dense + BM25 Sparse)
        FlashRank_Rerank: FlashRank Cross-Encoder Context Reranking
        Critic_CRAG: CriticAgent (CRAG Document Relevance Grading)
        Planner_Rewrite: PlannerAgent (Feedback Query Reformulation)

        Planner_Decompose --> Research_Hybrid
        Research_Hybrid --> FlashRank_Rerank
        FlashRank_Rerank --> Critic_CRAG

        state crag_decision <<choice>>
        Critic_CRAG --> crag_decision
        crag_decision --> Pass: Confidence >= 0.4
        crag_decision --> Low_Confidence: Irrelevant / Low Context
        Low_Confidence --> Planner_Rewrite: Query rewrite with Critic feedback
        Planner_Rewrite --> Research_Hybrid: Retry retrieval (max 1 loop)
    }

    DIRECT_CHAT --> Synthesis_Agent
    WEB_SEARCH --> Synthesis_Agent
    Pass --> Synthesis_Agent

    state Synthesis_Agent {
        diagram_extraction: Detect & extract architectural diagrams from metadata
        context_compression: Assemble verified context chunks & system instructions
    }

    Synthesis_Agent --> Resilient_LLM_Engine

    state Resilient_LLM_Engine {
        rate_limiter: Sliding-Window Token-Bucket RPM/TPM Check
        primary_model: Primary Gemini 2.0 / 2.5 Flash Generation
        fallback_cascade: Transparent Groq / OpenRouter / Ollama Failover on 429
        primary_model --> fallback_cascade: Fallback Triggered
    }

    state generation_mode <<choice>>
    Resilient_LLM_Engine --> generation_mode
    generation_mode --> SSE_Streaming: req.stream == True (Fast Real-Time Tokens)
    generation_mode --> Non_Streaming: req.stream == False (Analytical Mode)

    state Non_Streaming {
        Critic_SelfRAG: CriticAgent (Faithfulness & Groundedness Reflection)
        state grounded_decision <<choice>>
        Critic_SelfRAG --> grounded_decision
        grounded_decision --> Grounded_Pass: Factual claims supported by context
        grounded_decision --> Regenerate: Hallucination detected
        Regenerate --> Resilient_LLM_Engine: Re-prompt with strict critique
    }

    SSE_Streaming --> Persistence_Layer
    Grounded_Pass --> Persistence_Layer

    state Persistence_Layer {
        db_persist: PostgreSQL (Save conversation turn, session & usage)
        cache_persist: Upstash Redis (Index semantic cache for sub-ms reuse)
    }

    Persistence_Layer --> END
    END --> [*]
```

---

## 4. Component Catalog and Responsibilities

### 4.1 Client and Presentation Layer

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **Web SPA Interface** | [`frontend/`](frontend/) | Production web client with dark Titanium theme, real-time citation cards, SSE streaming event reader, model switcher, session history drawer, diagram viewer, and dynamic suggested follow-ups. | React 19, Tailwind CSS v4, Vite 8, EventSource API, Framer Motion |
| **Streamlit Dashboard** | [`streamlit_app/`](streamlit_app/) | Analytical interface offering model telemetry, chat history inspection, vector store exploration, document ingestion, and quota tracking. | Streamlit 1.40+, Custom CSS, Threading Queue Bridge |

### 4.2 API and Gateway Layer

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **FastAPI Core App** | [`backend/main.py`](backend/main.py) | Application entry point; initializes CORS, registers security headers, manages database lifespan, mounts routers, and serves static SPA fallbacks. | FastAPI 0.115+, Starlette, Uvicorn |
| **Anti-DDoS Rate Limiter** | [`backend/app/core/rate_limiter.py`](backend/app/core/rate_limiter.py) | High-speed in-memory sliding-window token bucket shield intercepting flood attacks (>12 req/5s) and scraper abuse at connection layer. | FastAPI BaseHTTPMiddleware, TokenBucket |
| **Security Headers** | [`backend/app/core/security.py`](backend/app/core/security.py) | Injects standard security HTTP headers into every response (nosniff, DENY, X-XSS-Protection, Referrer-Policy). | Starlette Middleware |
| **Chat API Routes** | [`backend/app/api/chat_routes.py`](backend/app/api/chat_routes.py) | Handles `/api/chat` (JSON / SSE stream with IP defense), `/api/scrape`, `/api/crawl`, and `/api/models`. | FastAPI APIRouter, StreamingResponse |
| **User & Quota Routes** | [`backend/app/api/user_routes.py`](backend/app/api/user_routes.py) | Handles `/api/user/identify`, `/api/user/quota`, `/api/sessions` (list & create), and `/api/sessions/{session_id}` (get & delete). | FastAPI APIRouter, Dependency Injection |
| **Frontend Router** | [`backend/app/api/frontend_routes.py`](backend/app/api/frontend_routes.py) | Serves the single-page application and static asset fallbacks for single-port deployments. | StaticFiles, FileResponse |

### 4.3 Application Core and Multi-Agent Orchestration

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **ChatService** | [`backend/app/services/chat_service.py`](backend/app/services/chat_service.py) | Business logic orchestrator; handles quota consumption, persists user questions/answers, checks semantic cache, and triggers agent streaming. | Python 3.11+, asyncio |
| **SupervisorAgent** | [`backend/app/agents/supervisor_agent.py`](backend/app/agents/supervisor_agent.py) | Orchestrator delegating workflow across specialized agents, tracking execution traces and telemetry. | Python 3.11+, Protocols |
| **RouterAgent** | [`backend/app/agents/router_agent.py`](backend/app/agents/router_agent.py) | Intent classification agent determining routing across direct conversation, document RAG, complex analytic, and web search. | Pydantic, Gemini Client |
| **PlannerAgent** | [`backend/app/agents/planner_agent.py`](backend/app/agents/planner_agent.py) | Sub-query decomposition agent and query rewriter responding to Critic feedback. | Pydantic, Gemini Client |
| **ResearchAgent** | [`backend/app/agents/research_agent.py`](backend/app/agents/research_agent.py) | Concurrent hybrid retrieval agent combining Qdrant dense vectors, BM25 sparse search, and FlashRank reranking. | FlashRank, Qdrant, asyncio |
| **CriticAgent** | [`backend/app/agents/critic_agent.py`](backend/app/agents/critic_agent.py) | Verification agent executing Corrective RAG document relevance grading and Self-RAG hallucination checking. | Pydantic, Gemini Client |
| **SynthesisAgent** | [`backend/app/agents/synthesis_agent.py`](backend/app/agents/synthesis_agent.py) | Assembles verified context passages, enforces document isolation directives, extracts diagrams, and prepares citations. | Pydantic, Python 3.11+ |
| **WebChatAgent** | [`backend/app/agents/chat_agent.py`](backend/app/agents/chat_agent.py) | Conversational façade unifying agent orchestration, synchronous/asynchronous streaming bridges, and vector store rehydration. | Python 3.11+, threading.Thread, queue.Queue |
| **UserService** | [`backend/app/services/user_service.py`](backend/app/services/user_service.py) | Manages dual-layer quota tracking (Client ID + IP address, 50 queries/day reset at UTC midnight). | SQLAlchemy, datetime |

### 4.4 RAG, Scraper, Memory and Vector Services

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **RAGService** | [`backend/app/services/rag_service.py`](backend/app/services/rag_service.py) | Document chunking, dense embedding generation, Qdrant Cloud hybrid ingestion, and Reciprocal Rank Fusion ($k=60$). | Qdrant Cloud, FAISS-CPU, rank-bm25 |
| **RerankService** | [`backend/app/services/rerank_service.py`](backend/app/services/rerank_service.py) | Local cross-encoder reranker refining top hybrid retrieval candidates down to the most relevant contexts. | FlashRank |
| **ScraperService** | [`backend/app/services/scraper_service.py`](backend/app/services/scraper_service.py) | 7-tier scraper cascade: Substack REST API, Medium Freedium + Apollo cache, News Googlebot referer spoofing, TLS JA3/JA4 impersonation, Trafilatura, Jina Reader, and Internet Archive. | Trafilatura, BeautifulSoup4, curl_cffi, Requests |
| **MemoryService** | [`backend/app/services/memory_service.py`](backend/app/services/memory_service.py) | Connects to Memori Labs; records conversation turns and extracts facts automatically into a semantic knowledge profile. | memori, SQLAlchemy |
| **QdrantService** | [`backend/app/services/qdrant_service.py`](backend/app/services/qdrant_service.py) | Cloud vector database hosting dense embeddings (3072-dim Cosine) and BM25 sparse vectors. | qdrant-client |

### 4.5 LLM Multi-Provider Fallback Cascade and Rate Limiter

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **ResilientLLMClient** | [`backend/app/clients/llm_client.py`](backend/app/clients/llm_client.py) | Cascades across 10 priority models across Google Gemini, Groq (LLaMA 3.3 70B), DeepSeek R1, and Ollama on rate limits or timeouts. | google-genai, groq |
| **Rate Limiter Core** | [`backend/app/core/rate_limiter.py`](backend/app/core/rate_limiter.py) | Thread-safe token buckets, sliding window RPM monitors, and exponential cooldown trackers. | threading.Lock, time |
| **GeminiClient** | [`backend/app/clients/gemini_client.py`](backend/app/clients/gemini_client.py) | Google GenAI SDK wrapper handling structured generation and real-time SSE streaming. | google-genai |
| **GroqClient** | [`backend/app/clients/groq_client.py`](backend/app/clients/groq_client.py) | Groq SDK client for fast failover models (LLaMA 3.3 70B). | groq |

### 4.6 Storage, Databases and Caches

| Store | Location / Mechanism | Schema / Structure |
| :--- | :--- | :--- |
| **Relational Database** | PostgreSQL on Aiven Cloud | Tables for users, sessions, messages, guest usage by IP/CID, and URL cache with SHA-256 hashes. |
| **Memori Labs Storage** | PostgreSQL tables | Tables for user entities, sessions, conversations, and knowledge graph facts. |
| **SemanticCacheService** | [`backend/app/cache/semantic_cache.py`](backend/app/cache/semantic_cache.py) on Upstash Cloud Redis | Two-tier semantic cache: Tier 1 instant SHA-256 exact match (<1ms) and Tier 2 dense vector cosine similarity (threshold >= 0.92) with local disk fallback. |
| **VectorStoreCache** | [`backend/app/cache/vector_cache.py`](backend/app/cache/vector_cache.py) (RAM + Disk) | In-memory LRU and persistent disk cache for FAISS indices and document metadata partitioned by URL hash. |
| **Cloud Vector Database** | [`backend/app/services/qdrant_service.py`](backend/app/services/qdrant_service.py) on Qdrant Cloud | Primary vector database hosting dense embeddings (3072-dim Cosine) and BM25 sparse vectors. Local FAISS acts as secondary backup. |

---

## 5. Security and Authentication Model

1. **Guest Device Tier**:
   - Devices are identified by a persistent UUID Client ID and their client IP address.
   - Guests receive 50 queries per day, tracked in PostgreSQL.
   - Both the Client ID and the client IP address are evaluated, preventing users from resetting their quota by clearing browser storage.
   - Quotas automatically reset daily at 00:00:00 UTC.

2. **HTTP Security Headers**:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `X-XSS-Protection: 1; mode=block`
   - `Referrer-Policy: strict-origin-when-cross-origin`

3. **Input Sanitization**:
   - Strips null bytes and control characters from queries and document inputs.
   - Detects and neutralizes prompt injection payloads trying to break document grounding.

---

## 6. Background Jobs, Concurrency and Threading Architecture

1. **Async and Sync Generator Streaming Bridge (`chat_agent.py`)**:
   - Streamlit operates in a synchronous execution thread per session, while LLM clients operate asynchronously via `asyncio`.
   - A dedicated daemon thread runs an isolated event loop, passing tokens, thinking steps, and citations into a thread-safe `queue.Queue`.
   - The caller consumes items from the queue with an active timeout, terminating on a sentinel token or raising cleanly on exceptions.

2. **Parallel Web and Domain Crawler (`scraper_service.py`)**:
   - Domain crawling and paywall bypass use an asynchronous worker pool with concurrency budgeting.
   - Internal link extraction prioritizes documentation, guides, and articles over social/utility links (`link_priority`).

3. **Pure React 19 Streaming Architecture (`frontend/src/App.jsx`)**:
   - Accumulates chunk tokens outside of React state updaters to prevent 2x token duplication during `<React.StrictMode>` development runs.
   - Features dynamic three-tier suggested follow-up questions engine combining document headings, assistant concepts, and query intent with history deduplication.
   - Dynamic environment URL resolution (`VITE_API_BASE_URL`) with automatic `/api` suffix normalization.

---

### Production Architectural Guarantees

- **Strictly Top-Level Imports**: Every Python module enforces static, top-level imports with zero inline imports.
- **Zero Namespace Collisions**: Modern namespace packaging across all backend domain packages.
- **Dedicated FAISS Disk Vector Store**: FAISS is persisted to disk under `data/vector_storage/`, while Qdrant operates in the cloud.
- **Start-to-End Exception Boundaries**: Functions and route handlers implement structured error handling with contextual logging.
