# WebChat AI — System Architecture & Technical Design

This document provides a comprehensive, production-grade architectural analysis of the **WebChat AI** platform based strictly on the actual codebase implementation.

---

## 1. High-Level System Architecture

WebChat AI provides a dual-interface conversational AI platform supporting both a **single-page web application (SPA)** and a **Streamlit analytical dashboard**. The backend is built with **FastAPI** and orchestrates an **Agentic RAG (Retrieval-Augmented Generation)** pipeline driven by **LangGraph**, utilizing hybrid retrieval (dense vectors + sparse BM25 + reciprocal rank fusion + cross-encoder reranking), multi-provider LLM fallback cascades (Google Gemini & Groq), persistent user semantic memory (Memori Labs), multi-tier paywall bypassing, and structured relational persistence via PostgreSQL/SQLite.

```mermaid
graph TB
    subgraph Client_Layer ["1. Client & Presentation Layer"]
        UI_SPA["Web SPA (Vanilla HTML5 / Modern CSS / JavaScript ES6+)<br/>• SSE Stream Consumer<br/>• Real-time Citation Drawer<br/>• Responsive Theme & Chat Drawer"]
        UI_Streamlit["Streamlit Dashboard (Python / Custom CSS)<br/>• Model Switcher & Quota Metrics<br/>• Document Ingestion & Chat Stream<br/>• Threading Queue Generator Bridge"]
    end

    subgraph API_Gateway ["2. API & Gateway Layer (FastAPI / Starlette)"]
        ASGI["Uvicorn ASGI Server (:8000)"]
        MW_CORS["CORSMiddleware (Configurable Origins)"]
        MW_Sec["SecurityHeadersMiddleware<br/>(nosniff, DENY, XSS, strict-origin)"]
        
        subgraph Routes ["API Routers"]
            R_Front["frontend_routes<br/>• / (SPA HTML)<br/>• /css, /js, /static"]
            R_Chat["chat_routes<br/>• POST /api/chat (JSON & SSE stream)<br/>• POST /api/scrape<br/>• POST /api/crawl<br/>• POST /api/upload-file<br/>• GET /api/models"]
            R_User["user_routes<br/>• POST /api/user/identify<br/>• GET /api/user/quota<br/>• GET /api/sessions<br/>• POST /api/sessions<br/>• GET /api/sessions/{id}<br/>• DELETE /api/sessions/{id}"]
        end
    end

    subgraph Core_Orchestration ["3. Application & Agentic Orchestration"]
        SVC_Chat["ChatService<br/>• Quota Verification<br/>• Session & Message CRUD<br/>• Semantic Cache Interception<br/>• Event Stream Formatting (SSE)"]
        AGENT_Chat["WebChatAgent<br/>• Async / Sync Streaming Bridge<br/>• LangGraph Orchestrator<br/>• Citation Formatting"]
        SVC_AgenticRAG["AgenticRAGService (LangGraph)<br/>• Routing & Sub-query Expansion<br/>• Corrective RAG (CRAG)<br/>• Self-RAG Groundedness Grader"]
        SVC_User["UserService<br/>• Guest Client ID Quota (5 Requests)<br/>• Registered Email Quota (50 Requests/Day)<br/>• UTC Midnight Quota Reset"]
    end

    subgraph Intelligence_RAG ["4. RAG, Retrieval & Memory Services"]
        SVC_RAG["RAGService<br/>• RecursiveCharacterTextSplitter (800 / 150)<br/>• GeminiEmbeddings (3072-dim)<br/>• HybridSearchIndex (FAISS + BM25 + RRF)"]
        SVC_Rerank["RerankService<br/>• FlashRank (ms-marco-TinyBERT-L-2-v2)"]
        SVC_Expand["QueryExpansionService<br/>• LLM Sub-query Decomposition"]
        SVC_Scraper["ScraperService<br/>• Trafilatura / BeautifulSoup<br/>• Jina Reader API / Archive.is Fallback<br/>• PyPDF Document Extractor<br/>• BFS Domain Crawler (ThreadPoolExecutor)"]
        SVC_Memory["MemoryService (Memori Labs)<br/>• LLM Lifecycle Registration<br/>• Entity Attribution (user_id)<br/>• Semantic Memory Recall<br/>• Background Memory Augmentation"]
        SVC_Qdrant["QdrantService<br/>• Cloud/Local Qdrant Collection (webchat_chunks)"]
    end

    subgraph LLM_Cascade ["5. Resilient Multi-Provider LLM Engine"]
        SVC_LLM["LLMService / ResilientLLMClient"]
        Limiter["Rate Limiting & Cooldown Protection<br/>• TokenBucket Rate Limiters<br/>• SlidingWindowTracker (RPM)<br/>• CooldownTracker (Exponential Backoff + Jitter)"]
        
        subgraph Providers ["Fallback Model Priority Chain"]
            M_Gemini["Google Gemini (google-genai SDK)<br/>• gemini-3.6-flash (Primary)<br/>• gemini-3.5-flash<br/>• gemini-3.5-flash-lite<br/>• gemini-3.1-flash-lite<br/>• gemini-2.5-flash"]
            M_Groq["Groq Cloud (groq SDK)<br/>• openai/gpt-oss-120b<br/>• groq/compound<br/>• openai/gpt-oss-20b<br/>• groq/compound-mini"]
        end
    end

    subgraph Storage_Layer ["6. Storage, Databases & Caches"]
        subgraph Relational_DB ["Relational Store (PostgreSQL / SQLite)"]
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

        subgraph Vector_Stores ["Vector Indices & Caches"]
            IDX_Qdrant["Qdrant Cloud Vector DB"]
            CACHE_DiskVector["Disk Vector Store - FAISS (data/vector_storage/)"]
            CACHE_Semantic["Disk Semantic Answer Cache (data/semantic_cache/)"]
        end
    end

    subgraph External_APIs ["7. External Services & APIs"]
        EXT_Google["Google Gemini AI Platform"]
        EXT_Groq["Groq Cloud Inference Engine"]
        EXT_DDG_Wiki["DuckDuckGo & Wikipedia APIs"]
        EXT_Jina["Jina Reader API (r.jina.ai)"]
        EXT_Wayback["Internet Archive Wayback Machine"]
    end

    %% Connections
    UI_SPA --> ASGI
    UI_Streamlit --> ASGI
    UI_Streamlit -.-> AGENT_Chat

    ASGI --> MW_CORS --> MW_Sec
    MW_Sec --> R_Front
    MW_Sec --> R_Chat
    MW_Sec --> R_User

    R_Chat --> SVC_Chat
    R_User --> SVC_User
    SVC_Chat --> SVC_User
    SVC_Chat --> AGENT_Chat
    AGENT_Chat --> SVC_AgenticRAG

    SVC_AgenticRAG --> SVC_RAG
    SVC_AgenticRAG --> SVC_Rerank
    SVC_AgenticRAG --> SVC_Expand
    SVC_AgenticRAG --> SVC_Memory
    SVC_AgenticRAG --> SVC_LLM
    SVC_AgenticRAG -.-> EXT_DDG_Wiki

    SVC_RAG --> IDX_FAISS
    SVC_RAG --> IDX_BM25
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

## 2. End-to-End Data & Request Flow

The diagram below maps the complete lifecycle of a chat request from the user interface down to storage, retrieval, generation, streaming, and automatic memory extraction.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant API as FastAPI Gateway (/api/chat)
    participant ChatSvc as ChatService
    participant UserSvc as UserService
    participant CacheSvc as SemanticCacheService
    participant Agent as WebChatAgent
    participant LangGraph as AgenticRAGService (LangGraph)
    participant HybridRAG as RAGService (FAISS + BM25)
    participant Memori as MemoryService (Memori Labs)
    participant LLMEngine as ResilientLLMClient
    participant DB as PostgreSQL / SQLite

    User->>API: POST /api/chat {query, url, model, session_id, stream: true}
    API->>ChatSvc: handle_chat_stream_async(req)
    
    %% Quota Verification
    ChatSvc->>UserSvc: check_and_consume_quota(email, client_id)
    UserSvc->>DB: Query / Increment webchat_users or webchat_guest_usage
    DB-->>UserSvc: Quota Approved
    UserSvc-->>ChatSvc: allowed=True, quota_info

    %% Attribution
    ChatSvc->>Memori: set_attribution(user_identifier)
    ChatSvc->>DB: append_message(role='user', content=query)

    %% Semantic Cache
    ChatSvc->>CacheSvc: get_cached_answer(query, context_hash)
    alt Exact Cache Hit
        CacheSvc-->>ChatSvc: cached_answer, citations
        ChatSvc-->>User: SSE Event: cached chunks + citations
    else Cache Miss
        ChatSvc->>Agent: answer_query_stream_events_async(...)
        
        %% LangGraph Pipeline
        Agent->>LangGraph: execute_agentic_rag_stream(...)
        
        LangGraph->>LangGraph: route_node (DIRECT_CHAT vs WEB_SEARCH vs DOCUMENT_RAG)
        alt Document RAG Path
            LangGraph->>LangGraph: decompose_and_expand_node (Sub-queries)
            LangGraph->>HybridRAG: hybrid_search (Dense FAISS + Sparse BM25 + RRF)
            HybridRAG-->>LangGraph: Retrieved Top-K Chunks
            LangGraph->>LangGraph: grade_documents_node (CRAG Relevance Grading)
            opt Low Relevance
                LangGraph->>LangGraph: transform_query_node -> web_search_node (DuckDuckGo)
            end
        end

        %% Context Synthesis & Memory Recall
        LangGraph->>Memori: get_relevant_memories(query, user_id)
        Memori-->>LangGraph: Recalled Facts / User Preferences
        LangGraph->>LangGraph: synthesize_context_node (Format Context + System Instructions)

        %% Ready Event
        LangGraph-->>Agent: yield {"type": "ready", context_chunks, prompt, system_instruction}
        Agent-->>User: SSE Event: reasoning steps + citations metadata

        %% LLM Generation with Fallback & Interception
        Agent->>LLMEngine: generate_stream_async(prompt, system_instruction)
        LLMEngine->>Memori: Native Memori Interception (Invoke.invoke)
        LLMEngine-->>User: SSE Event: token chunks (Streaming)

        %% Automatic Memory Capture & Caching
        LLMEngine->>Memori: Auto-capture conversation turn into memori_conversation_message & memori_entity_fact
        Agent-->>ChatSvc: Stream Finished (accumulated text)
        ChatSvc->>CacheSvc: set_cached_answer(query, answer, citations)
        ChatSvc->>DB: append_message(role='assistant', content=answer, citations)
        ChatSvc-->>User: SSE Event: [DONE]
    end
```

---

## 3. Detailed AI & Agentic RAG Pipeline (LangGraph Workflow)

The retrieval engine is built on **LangGraph** (`AgenticRAGService`), incorporating **Corrective RAG (CRAG)**, **Self-RAG (Faithfulness/Hallucination grading)**, and **Hybrid RRF Search**.

```mermaid
stateDiagram-v2
    [*] --> START
    START --> route_node: User Query Input

    state route_node <<choice>>
    note right of route_node
        Classifies query intent:
        • Direct greeting / conversational
        • Web search trigger keywords
        • Ingestion Document RAG
    end note

    route_node --> DIRECT_CHAT: Simple greeting / Conversational
    route_node --> WEB_SEARCH: Explicit web search request
    route_node --> DOCUMENT_RAG: Document analysis question

    state DIRECT_CHAT {
        synthesize_direct: Synthesize direct prompt
    }

    state WEB_SEARCH {
        web_search_direct: DuckDuckGo Web Search
    }

    state DOCUMENT_RAG {
        decompose: decompose_and_expand_node<br/>(Split complex comparative queries)
        retrieve: retrieve_node<br/>(Dense FAISS + Sparse BM25 + FlashRank Rerank)
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

## 4. Component Catalog & Responsibilities

### 4.1 Client & Presentation Layer

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **Web SPA Interface** | [frontend/](file:///d:/Projects/Personal-Projects/webchat/frontend/) | Ultra-responsive, production-grade web client with glassmorphic dark theme, real-time citation cards, SSE streaming event reader, model selection switcher, session history drawer, and quick ingestion buttons. | HTML5, Vanilla CSS (`index.css`), Vanilla JavaScript (`app.js`), EventSource API |
| **Streamlit Dashboard** | [streamlit_app/](file:///d:/Projects/Personal-Projects/webchat/streamlit_app/) | Full-featured secondary analytical interface offering real-time model telemetry, chat history inspection, manual vector store exploration, document ingestion, and quota dials. | Streamlit 1.40+, Custom CSS (`styles.py`), Threading Queue Bridge |

---

### 4.2 API & Gateway Layer

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **FastAPI Core App** | [backend/app/main.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/main.py) | Application entry point; initializes CORS, registers security headers, manages lifespan (DB migration + Memori storage build), and mounts routers. | FastAPI 0.115+, Starlette, Uvicorn |
| **Security Headers** | [security.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/core/security.py) | Injects standard security HTTP headers into every response (`nosniff`, `DENY`, `X-XSS-Protection`, `Referrer-Policy`) and sanitizes inputs. | Starlette Middleware |
| **Chat API Routes** | [chat_routes.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/api/chat_routes.py) | Handles `/api/chat` (JSON / SSE stream), `/api/scrape`, `/api/crawl`, `/api/upload-file`, and `/api/models`. | FastAPI APIRouter, StreamingResponse |
| **User & Quota Routes**| [user_routes.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/api/user_routes.py) | Handles `/api/user/identify`, `/api/user/quota`, `/api/sessions` (list & create), and `/api/sessions/{session_id}` (get & delete). | FastAPI APIRouter, Dependency Injection |
| **Frontend Router** | [frontend_routes.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/api/frontend_routes.py) | Serves the single-page application and static asset fallbacks. | StaticFiles, FileResponse |

---

### 4.3 Application Core & Orchestration

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **ChatService** | [chat_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/chat_service.py) | Business logic orchestrator; handles quota consumption, persists user questions/answers, checks semantic cache, triggers agent streaming, and sets memory attribution. | Python 3.11+, asyncio |
| **WebChatAgent** | [chat_agent.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/agents/chat_agent.py) | Coordinates Agentic RAG graph streaming, citation extraction, token yield formatting, and synchronous queue bridge for Streamlit. | asyncio, queue, threading |
| **AgenticRAGService** | [agentic_rag_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/agentic_rag_service.py) | Official LangGraph workflow executing routing, multi-hop query decomposition, hybrid retrieval, CRAG document grading, query rewriting, web search fallback, and Self-RAG evaluation. | LangGraph 0.2+, LangChain Core |
| **UserService** | [user_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/user_service.py) | Manages guest tracking (UUID client ID, 5 lifetime requests) and authenticated user daily quotas (50 requests/day, UTC midnight reset). | SQLAlchemy, datetime |

---

### 4.4 RAG, Scraper, Memory & Vector Services

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **RAGService** | [rag_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/rag_service.py) | Document chunking (`RecursiveCharacterTextSplitter`), dense embedding (`GeminiEmbeddings` with 3072 dims), sparse indexing (`BM25Retriever`), and Reciprocal Rank Fusion (`HybridSearchIndex`). | FAISS-CPU, rank-bm25, LangChain |
| **RerankService** | [rerank_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/rerank_service.py) | High-speed local cross-encoder reranker refining top hybrid retrieval candidates. | FlashRank (`ms-marco-TinyBERT-L-2-v2`) |
| **ScraperService** | [scraper_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/scraper_service.py) | Ingests URLs and PDFs; bypasses soft paywalls using multi-tier fallback: Trafilatura → BeautifulSoup → Jina Reader API → Internet Archive Wayback Machine. | Trafilatura, BeautifulSoup4, PyPDF, Requests |
| **MemoryService** | [memory_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/memory_service.py) | Connects to Memori Labs (Cloud or BYODB PostgreSQL); registers LLM client so conversation turns and facts are captured and recalled automatically. | `memori>=3.3.0`, SQLAlchemy |
| **QdrantService** | [qdrant_service.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/services/qdrant_service.py) | Connects to optional remote Qdrant vector database (`webchat_chunks` collection) with 3072-dimensional cosine similarity. | `qdrant-client>=1.19.0` |

---

### 4.5 LLM Multi-Provider Fallback Cascade & Rate Limiter

| Component | Path | Responsibility | Technologies / Dependencies |
| :--- | :--- | :--- | :--- |
| **ResilientLLMClient** | [llm_client.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/clients/llm_client.py) | Transparently cascades through 10 priority models across Google Gemini and Groq if rate limits (HTTP 429) or timeouts occur. | google-genai, groq |
| **Rate Limiter Core** | [rate_limiter.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/core/rate_limiter.py) | Thread-safe token buckets, sliding window RPM monitors, and exponential cooldown trackers preventing provider throttling. | threading.Lock, time, random |
| **GeminiClient** | [gemini_client.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/clients/gemini_client.py) | Official Google GenAI SDK wrapper registered with Memori for automatic memory capture. | `google-genai>=1.0.0` |
| **GroqClient** | [groq_client.py](file:///d:/Projects/Personal-Projects/webchat/backend/app/clients/groq_client.py) | Official Groq SDK client for fast failover models (`gpt-oss-120b`, `compound`). | `groq>=0.13.0` |

---

### 4.6 Storage, Databases & Caches

| Store | Location / Mechanism | Schema / Structure |
| :--- | :--- | :--- |
| **Relational Database** | PostgreSQL (`psycopg2-binary`) or SQLite (`webchat.db`) | • `webchat_users` (id, email, daily_requests_count, last_request_date)<br/>• `webchat_sessions` (session_id, user_id, guest_client_id, title, url)<br/>• `webchat_messages` (id, session_id, role, content, citations_json)<br/>• `webchat_guest_usage` (client_id, request_count)<br/>• `webchat_url_cache` (url_hash, url, title, content, vector_session_id) |
| **Memori Labs BYODB** | PostgreSQL (`memori_*` tables) | • `memori_entity` (user mapping)<br/>• `memori_session` (session links)<br/>• `memori_conversation` & `memori_conversation_message`<br/>• `memori_entity_fact` & `memori_knowledge_graph` |
| **Semantic Cache** | `data/semantic_cache/*.json` | JSON files containing SHA-256 hashed queries, normalized text, answers, and citations with 7-day TTL. |
| **Vector Store Cache** | `data/vector_storage/<hash>/` | Serialized FAISS indices and document chunk pickles stored per URL hash. |

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

## 7. Process & Execution Architecture (`backend/main.py`)

The application is architected around a unified execution entry point defined in [backend/main.py](file:///d:/Projects/Personal-Projects/webchat/backend/main.py). It detects execution arguments and launches either the FastAPI ASGI server or the Streamlit web dashboard, while re-exporting the ASGI `app` instance for production servers.

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

    subgraph Backend_Subsystems ["FastAPI Application Subsystems (backend/main.py)"]
        Lifespan["Application Lifespan<br/>• Database schema initialization (init_db)<br/>• Memori Labs BYODB storage build<br/>• Semantic & vector cache pre-warming"]
        Static_Mounts["Static Asset Mounts<br/>• /css -> frontend/css<br/>• /js -> frontend/js<br/>• /static -> frontend/"]
        Routes_Mount["API Routers<br/>• /api/chat (JSON & SSE stream)<br/>• /api/scrape & /api/crawl<br/>• /api/upload-file<br/>• /api/user/* & /api/sessions/*<br/>• / (SPA HTML index)"]
        
        Backend_Proc --> Lifespan
        Lifespan --> Static_Mounts
        Lifespan --> Routes_Mount
    end

    subgraph Storage_Backends ["Persistent Storage Backends"]
        Local_Disk["Local Filesystem (data/)<br/>• data/vector_storage/<hash>/ (FAISS on-disk vector store)<br/>• data/semantic_cache/*.json (7-day TTL)"]
        External_DB[("PostgreSQL Database (Aiven)<br/>Configured via DB_USER, DB_PASSWORD, DB_HOST, DB_NAME<br/>• Application relational tables<br/>• Memori Labs BYODB tables")]
        External_Qdrant[("Remote Qdrant Cluster (Qdrant Cloud)<br/>Configured via QDRANT_URL, QDRANT_API_KEY")]
    end

    subgraph External_APIs ["External Network Services"]
        API_Gemini["Google Gemini AI API (google-genai SDK)"]
        API_Groq["Groq Cloud API (groq SDK)"]
        API_DDG_Wiki["DuckDuckGo Instant Answer & Wikipedia APIs"]
        API_Jina["Jina Reader API (r.jina.ai)"]
        API_Wayback["Internet Archive Wayback Machine"]
    end

    Routes_Mount --> Local_Disk
    Routes_Mount --> External_DB
    Routes_Mount --> External_Qdrant
    Routes_Mount --> External_APIs
    Streamlit_Proc -.->|Local In-Memory / Agent Bridge| Routes_Mount
```

### Production Architectural Guarantees
- **Strictly Top-Level Imports**: Every Python module enforces static, top-level imports with zero inline imports or guarded `try...except` import blocks.
- **Zero Namespace Collisions**: Modern namespace packaging without legacy `__init__.py` files across all backend domain packages.
- **Dedicated FAISS Disk Vector Store**: FAISS is the sole vector index persisted to disk (`data/vector_storage/`), while Qdrant operates purely in the cloud without local disk fallbacks.
- **Start-to-End Exception Boundaries**: Every function and route handler implements structured `try...except` boundaries with contextual logging.


