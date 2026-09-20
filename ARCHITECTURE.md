# WebChat AI — System Architecture & Technical Design

This document provides a comprehensive, production-grade architectural analysis of the **WebChat AI** platform based strictly on the actual codebase implementation.

---

## 1. High-Level System Architecture

WebChat AI provides a dual-interface conversational AI platform supporting both a **single-page web application (SPA)** and a **Streamlit analytical dashboard**. The backend is built with **FastAPI** and orchestrates an **Agentic RAG (Retrieval-Augmented Generation)** pipeline driven by **LangGraph**, utilizing hybrid retrieval (dense vectors + sparse BM25 + reciprocal rank fusion + cross-encoder reranking), multi-provider LLM fallback cascades (Google Gemini & Groq), persistent user semantic memory (Memori Labs), multi-tier paywall bypassing, and structured relational persistence via PostgreSQL/SQLite.

```mermaid
graph TB
    subgraph Client_Layer ["1. Client & Presentation"]
        UI_SPA["React 19 SPA (SSE Streaming + KaTeX)"]
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
        SVC_Scraper["Multi-Engine Scraper (Trafilatura + Jina + BS4)"]
        SVC_Memory["Semantic Memory (Memori Labs)"]
    end

    subgraph LLM_Cascade ["5. Resilient LLM Engine (10-Tier Failover)"]
        SVC_LLM["Resilient LLM Client (Token-Bucket Limiter)"]
        M_Gemini["Google Gemini 2.5 / 3.6 Flash (Primary)"]
        M_Groq["Groq LLaMA 3.3 70B (Fallback)"]
        M_Ollama["Ollama / DeepSeek / OpenRouter (Cascades)"]
    end

    subgraph Storage_Layer ["6. Storage & Caches"]
        DB_Relational[("PostgreSQL Database (Aiven Cloud)")]
        DB_Vector[("Qdrant Cloud / FAISS Vector Store")]
        CACHE_Redis[("Upstash Cloud Redis (Two-Tier Semantic Cache)")]
    end

    %% Client & Gateway Connections
    UI_SPA --> FastAPI
    UI_Streamlit --> FastAPI
    FastAPI --> DDoS
    DDoS --> Quota
    Quota --> CACHE_Redis

    %% Cache Miss to Multi-Agent Supervisor
    CACHE_Redis -->|Cache Miss| Supervisor
    Supervisor --> Router
    Supervisor --> Planner
    Supervisor --> Researcher
    Supervisor --> Critic
    Supervisor --> Synthesizer

    %% Intelligence Layer
    Researcher --> SVC_RAG
    Researcher --> SVC_Rerank
    Supervisor --> SVC_Scraper
    Supervisor --> SVC_Memory

    %% LLM Cascade
    Synthesizer --> SVC_LLM
    SVC_LLM --> M_Gemini
    SVC_LLM --> M_Groq
    SVC_LLM --> M_Ollama

    %% Storage & Persistence Layer
    CACHE_Redis -->|Tier 1 Exact & Tier 2 Vector| UI_SPA
    Supervisor --> DB_Relational
    SVC_RAG --> DB_Vector
    SVC_Memory --> DB_Relational
```

---

### 2. End-to-End Data & Request Flow

The diagram below maps the complete lifecycle of a chat request from the client interface down to security middleware, quota checks, multi-agent orchestration, hybrid retrieval, streaming generation, and persistence.

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

        %% Dual-Layer Quota Verification
        ChatSvc->>UserSvc: check_and_consume_quota(client_id, ip_address)
        UserSvc->>DB: Check & increment quota in webchat_guest_usage
        DB-->>UserSvc: Quota Approved (50 daily limit per IP + CID)
        UserSvc-->>ChatSvc: allowed=True, quota_info

        %% Semantic Cache Lookup
        ChatSvc->>CacheSvc: get_cached_answer(query, context_hash)
        alt Tier 1 / Tier 2 Cache Hit (Similarity >= 0.92)
            CacheSvc-->>ChatSvc: cached_answer, citations
            CacheSvc-->>User: SSE Event: instant response + citations
        else Cache Miss
            ChatSvc->>Supervisor: orchestrate_stream_async(query, vector_store, ...)

            %% Multi-Agent Pipeline
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

            %% Resilient LLM Streaming
            ChatSvc->>LLMEngine: generate_stream_async(prompt, system_instruction)
            loop Token Generation
                LLMEngine-->>ChatSvc: token chunk
                ChatSvc-->>User: SSE Event: text chunk
            end

            %% Persistence
            ChatSvc->>DB: append_message(role='user', content=query)
            ChatSvc->>DB: append_message(role='assistant', content=answer, citations)
            ChatSvc->>CacheSvc: set_cached_answer(query, answer, citations)
            ChatSvc-->>User: SSE Event: [DONE]
        end
    end
```

---

## 3. Detailed AI & Agentic RAG Pipeline (Multi-Agent Supervisor Workflow)

The core intelligence layer operates as an asynchronous multi-agent supervisor architecture implementing **Corrective RAG (CRAG)**, **Self-RAG reflection**, and **Parallel Hybrid Retrieval**.

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
        primary_model: Primary Gemini 2.5 / 3.6 Flash Generation
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

## 4. Component Catalog & Responsibilities

### 4.1 Client & Presentation Layer

| Component               | Path                                                                           | Responsibility                                                                                                                                                                                                   | Technologies / Dependencies                                                      |
| :---------------------- | :----------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **Web SPA Interface**   | [frontend/](file:///d:/Projects/Personal-Projects/webchat/frontend/)           | Ultra-responsive, production-grade web client with glassmorphic dark theme, real-time citation cards, SSE streaming event reader, model selection switcher, session history drawer, and quick ingestion buttons. | HTML5, Vanilla CSS (`index.css`), Vanilla JavaScript (`app.js`), EventSource API |
| **Streamlit Dashboard** | [streamlit_app/](file:///d:/Projects/Personal-Projects/webchat/streamlit_app/) | Full-featured secondary analytical interface offering real-time model telemetry, chat history inspection, manual vector store exploration, document ingestion, and quota dials.                                  | Streamlit 1.40+, Custom CSS (`styles.py`), Threading Queue Bridge                |

---

### 4.2 API & Gateway Layer

| Component                  | Path                                                                                                   | Responsibility                                                                                                                                     | Technologies / Dependencies             |
| :------------------------- | :----------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------- |
| **FastAPI Core App**       | [backend/app/main.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/main.py)               | Application entry point; initializes CORS, registers security headers, manages lifespan (DB migration + Memori storage build), and mounts routers. | FastAPI 0.115+, Starlette, Uvicorn      |
| **Anti-DDoS Rate Limiter** | [rate_limiter.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/core/rate_limiter.py)      | High-speed in-memory sliding-window token bucket shield intercepting flood attacks (>12 req/5s) and scraper abuse at the connection layer.         | FastAPI BaseHTTPMiddleware, TokenBucket |
| **Security Headers**       | [security.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/core/security.py)              | Injects standard security HTTP headers into every response (`nosniff`, `DENY`, `X-XSS-Protection`, `Referrer-Policy`) and sanitizes inputs.        | Starlette Middleware                    |
| **Chat API Routes**        | [chat_routes.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/api/chat_routes.py)         | Handles `/api/chat` (JSON / SSE stream with IP defense), `/api/scrape`, `/api/crawl`, and `/api/models`.                                           | FastAPI APIRouter, StreamingResponse    |
| **User & Quota Routes**    | [user_routes.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/api/user_routes.py)         | Handles `/api/user/identify`, `/api/user/quota`, `/api/sessions` (list & create), and `/api/sessions/{session_id}` (get & delete).                 | FastAPI APIRouter, Dependency Injection |
| **Frontend Router**        | [frontend_routes.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/api/frontend_routes.py) | Serves the single-page application and static asset fallbacks.                                                                                     | StaticFiles, FileResponse               |

---

### 4.3 Application Core & Orchestration

| Component             | Path                                                                                                                | Responsibility                                                                                                                                                                        | Technologies / Dependencies    |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------- |
| **ChatService**       | [chat_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/chat_service.py)               | Business logic orchestrator; handles quota consumption, persists user questions/answers, checks semantic cache, triggers agent streaming, and sets memory attribution.                | Python 3.11+, asyncio          |
| **WebChatAgent**      | [chat_agent.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/agents/chat_agent.py)                     | Coordinates Agentic RAG graph streaming, citation extraction, token yield formatting, and synchronous queue bridge for Streamlit.                                                     | asyncio, queue, threading      |
| **AgenticRAGService** | [agentic_rag_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/agentic_rag_service.py) | Official LangGraph workflow executing routing, multi-hop query decomposition, hybrid retrieval, CRAG document grading, query rewriting, web search fallback, and Self-RAG evaluation. | LangGraph 0.2+, LangChain Core |
| **UserService**       | [user_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/user_service.py)               | Manages guest tracking (UUID client ID, 5 lifetime requests) and authenticated user daily quotas (50 requests/day, UTC midnight reset).                                               | SQLAlchemy, datetime           |

---

### 4.4 RAG, Scraper, Memory & Vector Services

| Component          | Path                                                                                                        | Responsibility                                                                                                                                                                                   | Technologies / Dependencies                   |
| :----------------- | :---------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| **RAGService**     | [rag_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/rag_service.py)         | Document chunking (`RecursiveCharacterTextSplitter`), dense embedding (`GeminiEmbeddings` with 3072 dims), Qdrant Cloud hybrid ingestion, and Reciprocal Rank Fusion (`MultiEngineHybridIndex`). | Qdrant Cloud, FAISS-CPU, rank-bm25, LangChain |
| **RerankService**  | [rerank_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/rerank_service.py)   | High-speed local cross-encoder reranker refining top hybrid retrieval candidates down to the most relevant contexts.                                                                             | FlashRank (`ms-marco-TinyBERT-L-2-v2`)        |
| **ScraperService** | [scraper_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/scraper_service.py) | Ingests URLs and PDFs; bypasses soft paywalls using multi-tier fallback: Trafilatura -> BeautifulSoup -> Jina Reader API -> Internet Archive Wayback Machine.                                    | Trafilatura, BeautifulSoup4, PyPDF, Requests  |
| **MemoryService**  | [memory_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/memory_service.py)   | Connects to Memori Labs (Cloud or BYODB PostgreSQL); registers LLM client so conversation turns and facts are captured and recalled automatically.                                               | `memori>=3.3.0`, SQLAlchemy                   |
| **QdrantService**  | [qdrant_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/qdrant_service.py)   | **Mandatory Primary Vector Database**: Connects to Qdrant Cloud cluster with 3072-dim dense cosine similarity, BM25 sparse vectors, and Binary Quantization (BQ) fused via RRF.                  | `qdrant-client>=1.19.0`                       |

---

### 4.5 LLM Multi-Provider Fallback Cascade & Rate Limiter

| Component              | Path                                                                                                   | Responsibility                                                                                                               | Technologies / Dependencies  |
| :--------------------- | :----------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| **ResilientLLMClient** | [llm_client.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/clients/llm_client.py)       | Transparently cascades through 10 priority models across Google Gemini and Groq if rate limits (HTTP 429) or timeouts occur. | google-genai, groq           |
| **Rate Limiter Core**  | [rate_limiter.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/core/rate_limiter.py)      | Thread-safe token buckets, sliding window RPM monitors, and exponential cooldown trackers preventing provider throttling.    | threading.Lock, time, random |
| **GeminiClient**       | [gemini_client.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/clients/gemini_client.py) | Official Google GenAI SDK wrapper registered with Memori for automatic memory capture.                                       | `google-genai>=1.0.0`        |
| **GroqClient**         | [groq_client.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/clients/groq_client.py)     | Official Groq SDK client for fast failover models (`gpt-oss-120b`, `compound`).                                              | `groq>=0.13.0`               |

---

### 4.6 Storage, Databases & Caches

| Store                            | Location / Mechanism                                                                | Schema / Structure                                                                                                                                                                                                                                                                                                                                              |
| :------------------------------- | :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Relational Database**          | PostgreSQL (`psycopg2-binary`) on Aiven Cloud                                       | - `webchat_users` (id, email, daily_requests_count, last_request_date)<br/>- `webchat_sessions` (session_id, user_id, guest_client_id, title, url)<br/>- `webchat_messages` (id, session_id, role, content, citations_json)<br/>- `webchat_guest_usage` (client_id, request_count)<br/>- `webchat_url_cache` (url_hash, url, title, content, vector_session_id) |
| **Memori Labs BYODB**            | PostgreSQL (`memori_*` tables)                                                      | - `memori_entity` (user mapping)<br/>- `memori_session` (session links)<br/>- `memori_conversation` & `memori_conversation_message`<br/>- `memori_entity_fact` & `memori_knowledge_graph`                                                                                                                                                                       |
| **Upstash Redis Semantic Cache** | Upstash Cloud Redis (`rediss://...`) + Local Disk Fallback (`data/semantic_cache/`) | Two-tier semantic cache: Tier 1 instant SHA-256 exact match (<1ms) + Tier 2 dense vector cosine similarity (threshold >= 0.92, Gemini 3072-dim embeddings) partitioned by URL context hash with configurable 7-day TTL (`SEMANTIC_CACHE_TTL`).                                                                                                                  |
| **Mandatory Cloud Vector DB**    | Qdrant Cloud Cluster (`QDRANT_URL`)                                                 | Mandatory primary vector database hosting dense embeddings (3072-dim Cosine), BM25 sparse vectors, and Binary Quantization (BQ) fused with Reciprocal Rank Fusion (RRF). Local FAISS disk cache serves as secondary backup.                                                                                                                                     |

---

## 5. Security & Authentication Model

1. **Guest Tier (Anonymous)**:
   - Clients are assigned a `guest_client_id` (UUID generated on client side and persisted in `localStorage` or session state).
   - Guests are granted **5 free requests** tracked in `webchat_guest_usage`.
   - Once 5 requests are exhausted, the server issues an HTTP `403 Forbidden` with payload `{"requires_email": true}`.
2. **Registered User Tier (Email Identification)**:
   - Users provide their email address via `/api/user/identify`.
   - Registered users receive **50 requests per day**, reset automatically at `00:00:00 UTC` by comparing `last_request_date` against `datetime.now(timezone.utc).strftime("%Y-%m-%d")`.
3. **HTTP Security Headers**:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `X-XSS-Protection: 1; mode=block`
   - `Referrer-Policy: strict-origin-when-cross-origin`
4. **Input Sanitization**:
   - Strip null bytes (`\x00`) and ASCII control characters from queries and document inputs.

---

## 6. Background Jobs, Concurrency & Threading Architecture

The system utilizes an asynchronous event-driven architecture coupled with managed thread pools and queues to handle long-running I/O without blocking:

1. **Async / Sync Generator Streaming Bridge (`chat_agent.py`)**:
   - Streamlit operates in a synchronous execution thread per session, while LangGraph and the LLM clients operate natively asynchronously via `asyncio`.
   - A dedicated daemon thread runs an isolated `asyncio.new_event_loop()`, passing tokens, thinking steps, and citations into a thread-safe `queue.Queue`.
   - The caller consumes items from the queue with an active timeout, terminating on a sentinel token (`None`) or raising cleanly on exceptions.

2. **Parallel Web & Domain Crawler (`scraper_service.py`)**:
   - Domain crawling and paywall bypass use `concurrent.futures.ThreadPoolExecutor(max_workers=min(len(urls), 6))`.
   - Scraping tasks run in parallel across worker threads, utilizing timeout budgets and session pools.

3. **Background Semantic Memory Ingestion (`memory_service.py`)**:
   - Memori Labs hooks into LLM invocations, executing knowledge graph entity/fact extraction asynchronously so response generation latency is not impacted.

---

### Production Architectural Guarantees

- **Strictly Top-Level Imports**: Every Python module enforces static, top-level imports with zero inline imports or guarded `try...except` import blocks.
- **Zero Namespace Collisions**: Modern namespace packaging without legacy `__init__.py` files across all backend domain packages.
- **Dedicated FAISS Disk Vector Store**: FAISS is the sole vector index persisted to disk (`data/vector_storage/`), while Qdrant operates purely in the cloud without local disk fallbacks.
- **Start-to-End Exception Boundaries**: Every function and route handler implements structured `try...except` boundaries with contextual logging.
