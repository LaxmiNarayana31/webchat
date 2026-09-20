import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.clients.gemini_client import GeminiClient, gemini_client
from backend.app.core.logging import logger
from backend.app.services.qdrant_service import qdrant_service
from backend.app.services.query_expansion_service import query_expansion_service
from backend.app.services.rerank_service import rerank_service


class GeminiEmbeddings(Embeddings):
    """Google Gemini embeddings wrapper using the unified GeminiClient."""

    def __init__(self, api_key: Optional[str] = None):
        """Initializes Gemini embeddings with API key and model selection."""
        try:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.client = GeminiClient(api_key=self.api_key) if api_key else gemini_client
            self.model = "gemini-embedding-2"
        except Exception as e:
            logger.error(f"Error initializing GeminiEmbeddings: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds multiple document texts into vector representations."""
        try:
            return self.client.embed_texts(texts, model=self.model)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return [[0.0] * 3072 for _ in texts]

    def embed_query(self, query: str) -> List[float]:  # type: ignore
        """Embeds a single query string into a vector representation."""
        try:
            embeddings = self.client.embed_texts([query], model=self.model)
            return embeddings[0] if embeddings else [0.0] * 3072
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 3072


class HybridSearchIndex:
    """Hybrid search index combining FAISS dense vectors, BM25 sparse keywords, and RRF."""

    def __init__(
        self,
        vector_store: FAISS,
        bm25_retriever: BM25Retriever,
        documents: List[Document],
    ):
        """Initializes hybrid search index with FAISS store and BM25 retriever."""
        try:
            self.vector_store = vector_store
            self.bm25_retriever = bm25_retriever
            self.documents = documents
        except Exception as e:
            logger.error(f"Error initializing HybridSearchIndex: {e}", exc_info=True)
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Provides direct FAISS-compatible interface while using hybrid search."""
        try:
            return self.hybrid_search(query=query, top_k=k)
        except Exception as e:
            logger.error(f"Error in HybridSearchIndex.similarity_search: {e}", exc_info=True)
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Executes hybrid dense and sparse retrieval with reciprocal rank fusion."""
        try:
            fetch_k = max(top_k * 2, 8)

            # Dense FAISS retrieval
            try:
                dense_docs = self.vector_store.similarity_search(query, k=fetch_k)
            except Exception as e:
                logger.warning(f"Dense FAISS search error ({e}). Falling back to sparse search.")
                dense_docs = []

            # Sparse BM25 retrieval
            try:
                sparse_docs = self.bm25_retriever.invoke(query)
                sparse_docs = sparse_docs[:fetch_k]
            except Exception as e:
                logger.warning(f"Sparse BM25 search error ({e}). Falling back to dense search.")
                sparse_docs = []

            # If both empty, return empty
            if not dense_docs and not sparse_docs:
                return []

            # Reciprocal Rank Fusion (RRF) & Reranking
            rrf_scores: Dict[int, float] = {}
            doc_lookup: Dict[int, Document] = {}

            # Dense ranking scores
            for rank, doc in enumerate(dense_docs, start=1):
                chunk_idx = doc.metadata.get("chunk_index", rank)
                doc_lookup[chunk_idx] = doc
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + (dense_weight / (rrf_k + rank))

            # Sparse BM25 ranking scores
            for rank, doc in enumerate(sparse_docs, start=1):
                chunk_idx = doc.metadata.get("chunk_index", rank)
                doc_lookup[chunk_idx] = doc
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + (bm25_weight / (rrf_k + rank))

            # Exact keyword match boost (reranking enhancement)
            query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
            for chunk_idx, doc in doc_lookup.items():
                if query_terms:
                    doc_text = doc.page_content.lower()
                    matched_terms = sum(1 for term in query_terms if term in doc_text)
                    if matched_terms > 0:
                        term_boost = (matched_terms / len(query_terms)) * 0.005
                        rrf_scores[chunk_idx] += term_boost

            # Sort documents by combined RRF score descending
            sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

            ranked_docs = [doc_lookup[idx] for idx in sorted_indices[:top_k]]
            return ranked_docs
        except Exception as e:
            logger.error(f"Error executing HybridSearchIndex.hybrid_search: {e}", exc_info=True)
            return []


class MultiEngineHybridIndex:
    """Multi-engine hybrid index prioritizing Qdrant with FAISS fallback."""

    def __init__(
        self,
        collection_name: str,
        faiss_hybrid_index: HybridSearchIndex,
        documents: List[Document],
        embeddings: GeminiEmbeddings,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initializes multi-engine hybrid index with Qdrant collection and FAISS fallback."""
        try:
            self.collection_name = collection_name
            self.faiss_hybrid_index = faiss_hybrid_index
            self.documents = documents
            self.embeddings = embeddings
            self.session_id = session_id
            self.metadata = metadata or {}
        except Exception as e:
            logger.error(f"Error initializing MultiEngineHybridIndex: {e}", exc_info=True)
            raise

    @property
    def vector_store(self) -> Optional[FAISS]:
        """Compatibility property for vector cache serialization."""
        try:
            if self.faiss_hybrid_index is not None:
                return self.faiss_hybrid_index.vector_store
            return None
        except Exception as e:
            logger.error(f"Error accessing vector_store property: {e}", exc_info=True)
            return None

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Executes similarity search via hybrid retrieval pipeline."""
        try:
            return self.hybrid_search(query=query, top_k=k)
        except Exception as e:
            logger.error(f"Error in MultiEngineHybridIndex.similarity_search: {e}", exc_info=True)
            return []

    def hybrid_search(self, query: str, top_k: int = 4) -> List[Document]:
        """Executes Qdrant hybrid retrieval with automatic fallback to FAISS."""
        try:
            # Attempt Qdrant Primary Search
            try:
                query_dense = self.embeddings.embed_query(query)
                qdrant_results = qdrant_service.search_hybrid(
                    collection_name=self.collection_name,
                    query_text=query,
                    query_dense_vector=query_dense,
                    top_k=top_k,
                    filter_session_id=self.session_id,
                )
                if qdrant_results:
                    docs: List[Document] = []
                    for res in qdrant_results:
                        docs.append(
                            Document(
                                page_content=res["content"],
                                metadata={
                                    "chunk_index": res.get("chunk_index", 0),
                                    "parent_id": res.get("parent_id", ""),
                                    "parent_text": res.get("parent_text", ""),
                                    "url": res.get("url", ""),
                                    "title": res.get("title", ""),
                                    "session_id": res.get("session_id", ""),
                                },
                            )
                        )
                    logger.info(f"MultiEngineHybridIndex: Retrieved {len(docs)} chunks from Primary Qdrant for session {self.session_id}.")
                    return docs
            except Exception as e:
                logger.warning(f"MultiEngineHybridIndex: Qdrant search encountered error: {e}. Failing over to FAISS.")

            # Secondary Fallback: FAISS + BM25 Hybrid Index
            if self.faiss_hybrid_index is not None:
                logger.info("MultiEngineHybridIndex: Executing Secondary Fallback (FAISS + BM25 RRF).")
                return self.faiss_hybrid_index.hybrid_search(query=query, top_k=top_k)
            return []
        except Exception as e:
            logger.error(f"Error in MultiEngineHybridIndex.hybrid_search: {e}", exc_info=True)
            return []


class RAGService:
    """RAG service for hierarchical chunking, indexing, hybrid retrieval, and prompt generation."""

    def __init__(self, api_key: Optional[str] = None):
        """Initializes RAG service with parent-child splitters and embeddings."""
        try:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.embeddings = GeminiEmbeddings(api_key=self.api_key)

            # Parent chunker for complete contextual comprehension (2,000 tokens)
            self.parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=250,
                separators=["\n\n", "\n", " ", ""],
            )

            # Child chunker for high-precision vector search (400 tokens)
            self.child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
        except Exception as e:
            logger.error(f"Error initializing RAGService: {e}", exc_info=True)
            raise

    def build_vectorstore(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[MultiEngineHybridIndex], Optional[str]]:
        """Constructs hierarchical parent-child vector index across Qdrant and FAISS."""
        try:
            if not text or not text.strip():
                return None, "Document content is empty."

            meta = metadata or {}

            # Parent Split (supports multi-page site crawls or single documents)
            if meta.get("pages"):
                raw_docs = []
                for p in meta["pages"]:
                    p_content = p.get("content", "")
                    if p_content and p_content.strip():
                        p_meta = dict(meta)
                        p_meta["url"] = p.get("url") or meta.get("url", "")
                        p_meta["title"] = p.get("title") or meta.get("title", "")
                        raw_docs.append(Document(page_content=p_content, metadata=p_meta))
                parent_docs = self.parent_splitter.split_documents(raw_docs) if raw_docs else []
            else:
                raw_doc = Document(page_content=text, metadata=meta)
                parent_docs = self.parent_splitter.split_documents([raw_doc])

            if not parent_docs:
                return None, "Failed to create parent chunks from document."

            child_docs: List[Document] = []
            chunk_counter = 0

            # Child Split per Parent with accurate source page attribution
            for p_idx, p_doc in enumerate(parent_docs):
                parent_id = f"p_{p_idx}"
                parent_text = p_doc.page_content

                sub_chunks = self.child_splitter.split_documents([p_doc])
                for c_doc in sub_chunks:
                    c_meta = dict(p_doc.metadata)
                    c_meta.update({
                        "chunk_index": chunk_counter,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "url": p_doc.metadata.get("url") or meta.get("url", ""),
                        "title": p_doc.metadata.get("title") or meta.get("title", ""),
                    })
                    c_doc.metadata = c_meta
                    child_docs.append(c_doc)
                    chunk_counter += 1

            # Add Multimodal Image Document Chunks if present
            images = meta.get("images") or []
            for img in images[:20]:
                img_url = img.get("url")
                img_alt = img.get("alt", "Content Diagram/Image")
                img_ctx = img.get("context", "")
                if img_url:
                    img_doc_content = (
                        f"[Visual Content Image Diagram / Illustration]\n"
                        f"Diagram Title: {img_alt}\n"
                        f"Image URL: {img_url}\n"
                        f"Caption & Architectural Context: {img_ctx or img_alt}"
                    )
                    img_meta = dict(meta)
                    img_meta.update({
                        "chunk_index": chunk_counter,
                        "parent_id": f"img_{chunk_counter}",
                        "parent_text": img_doc_content,
                        "is_image": True,
                        "image_url": img_url,
                        "url": meta.get("url", ""),
                        "title": meta.get("title", ""),
                    })
                    child_docs.append(Document(page_content=img_doc_content, metadata=img_meta))
                    chunk_counter += 1

            if not child_docs:
                child_docs = parent_docs

            child_texts = [c.page_content for c in child_docs]
            child_metas = [c.metadata for c in child_docs]

            # Use a single global collection for all web chat sources
            collection_name = "web-chat"

            # Embed & Insert into Qdrant Primary
            dense_embeddings = self.embeddings.embed_documents(child_texts)
            qdrant_service.insert_hybrid_chunks(
                collection_name=collection_name,
                texts=child_texts,
                dense_embeddings=dense_embeddings,
                metadatas=child_metas,
            )

            # Build Secondary FAISS Dense Vector Store + BM25 Retriever
            faiss_vs = FAISS.from_documents(child_docs, self.embeddings)
            text_embeddings = list(zip(child_texts, dense_embeddings))
            faiss_vs = FAISS.from_embeddings(text_embeddings, self.embeddings, metadatas=child_metas)
            bm25_retriever = BM25Retriever.from_documents(child_docs, k=len(child_docs))
            faiss_hybrid = HybridSearchIndex(
                vector_store=faiss_vs,
                bm25_retriever=bm25_retriever,
                documents=child_docs,
            )

            # MultiEngine Hybrid Index encapsulation
            multi_engine_index = MultiEngineHybridIndex(
                collection_name=collection_name,
                faiss_hybrid_index=faiss_hybrid,
                documents=child_docs,
                embeddings=self.embeddings,
                session_id=meta.get("session_id"),
                metadata=meta,
            )

            logger.info(
                f"RAGService: Hierarchical Index Built ({len(parent_docs)} parents, {len(child_docs)} child chunks, Qdrant Primary + FAISS Secondary)."
            )
            return multi_engine_index, None
        except Exception as e:
            logger.error(f"Error creating Multi-Engine search index: {e}", exc_info=True)
            return None, str(e)

    def get_session_index(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultiEngineHybridIndex:
        """Retrieves session-bound hybrid index from Qdrant collection."""
        try:
            return MultiEngineHybridIndex(
                collection_name="web-chat",
                faiss_hybrid_index=None,  # type: ignore
                documents=[],
                embeddings=self.embeddings,
                session_id=session_id,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Error creating session index: {e}", exc_info=True)
            raise

    def retrieve_context(
        self,
        index_or_store: Union[MultiEngineHybridIndex, HybridSearchIndex, FAISS, Any],
        query: str,
        top_k: int = 4,
        use_expansion: bool = True,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieves and reranks parent context chunks matching user query."""
        try:
            if not index_or_store:
                return []

            # Query Expansion (HyDE + Multi-Query)
            search_queries = [query]
            if use_expansion:
                try:
                    expanded = query_expansion_service.expand_query(query)
                    for eq in expanded:
                        if eq not in search_queries:
                            search_queries.append(eq)
                except Exception as ex_err:
                    logger.warning(f"Query expansion warning: {ex_err}")

            # Multi-Query Retrieval
            candidate_docs: List[Document] = []
            seen_chunk_indices: Set[int] = set()

            fetch_per_query = max(top_k * 2, 6)
            for q in search_queries:
                if isinstance(index_or_store, (MultiEngineHybridIndex, HybridSearchIndex)):
                    docs = index_or_store.hybrid_search(query=q, top_k=fetch_per_query)
                elif hasattr(index_or_store, "hybrid_search"):
                    docs = index_or_store.hybrid_search(query=q, top_k=fetch_per_query)
                elif hasattr(index_or_store, "similarity_search"):
                    docs = index_or_store.similarity_search(q, k=fetch_per_query)
                else:
                    docs = []

                for doc in docs:
                    c_idx = doc.metadata.get("chunk_index", hash(doc.page_content))
                    if c_idx not in seen_chunk_indices:
                        seen_chunk_indices.add(c_idx)
                        candidate_docs.append(doc)

            if not candidate_docs:
                return []

            candidate_dicts = [
                {
                    "content": d.page_content,
                    "chunk_index": d.metadata.get("chunk_index", 0),
                    "parent_id": d.metadata.get("parent_id", ""),
                    "parent_text": d.metadata.get("parent_text", d.page_content),
                    "url": d.metadata.get("url", ""),
                    "title": d.metadata.get("title", ""),
                }
                for d in candidate_docs
            ]

            # Local Neural Cross-Encoder Reranking
            if use_rerank and len(candidate_dicts) > 1:
                reranked = rerank_service.rerank(
                    query=query,
                    documents=candidate_dicts,
                    top_k=len(candidate_dicts),
                )
            else:
                reranked = candidate_dicts

            # Parent Resolution & Deduplication
            parent_resolved: List[Dict[str, Any]] = []
            seen_parent_ids: Set[str] = set()

            for item in reranked:
                p_id = item.get("parent_id")
                if p_id and p_id in seen_parent_ids:
                    continue
                if p_id:
                    seen_parent_ids.add(p_id)

                # Use full parent context for the prompt
                parent_content = item.get("parent_text") or item.get("content", "")
                parent_resolved.append({
                    "content": parent_content,
                    "chunk_index": item.get("chunk_index", 0),
                    "parent_id": p_id or "",
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "rerank_score": item.get("rerank_score", 0.0),
                })

                if len(parent_resolved) >= top_k:
                    break

            logger.info(
                f"RAGService: Context retrieved ({len(parent_resolved)} parent windows from {len(candidate_dicts)} candidates, queries={len(search_queries)})."
            )
            return parent_resolved

        except Exception as e:
            logger.error(f"Error retrieving context in RAGService: {e}", exc_info=True)
            return []

    def format_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_memories: Optional[List[str]] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Formats grounded prompt with context chunks, chat history, and system instructions."""
        try:
            doc_title = ""
            doc_url = ""
            if document_metadata:
                doc_title = document_metadata.get("title", "")
                doc_url = document_metadata.get("url", "")

            if not doc_title and context_chunks:
                doc_title = context_chunks[0].get("title", "")
            if not doc_url and context_chunks:
                doc_url = context_chunks[0].get("url", "")

            doc_header = ""
            if doc_title or doc_url:
                doc_header = "Active Document Context:\n"
                if doc_title:
                    doc_header += f"- Title: {doc_title}\n"
                if doc_url:
                    doc_header += f"- Source URL: {doc_url}\n"
                doc_header += "\n"

            system_instruction = (
                "You are WebChat, an advanced AI research assistant specialized in analyzing and answering questions about the provided website or document.\n\n"
                "STRICT CONTEXT GROUNDING & BEHAVIORAL GUIDELINES:\n"
                "1. STRICT CONTEXTUAL ACCURACY & FAITHFUL ATTRIBUTION: Ground your answers strictly and solely in the provided Document Context and Context Excerpts. "
                "Never invent facts, speculate beyond what is documented, or bring in outside knowledge about external entities, people, or events. "
                "Accurately represent the document's depth on any given topic: DO NOT imply or state that the document contains an extensive, dedicated guide or tutorial "
                "if the document only mentions the topic in passing, as a sub-point, or as one element of a broader strategy (e.g. mentioning hybrid search as one retrieval strategy). "
                "Explicitly distinguish what the active document specifically states versus what is broader technical context.\n"
                "2. PRECISE TECHNICAL DEFINITIONS & ARCHITECTURAL DISTINCTIONS: When explaining technical concepts mentioned in the text (such as Hybrid RAG, semantic search, BM25, or reranking), "
                "provide precise, industry-standard explanations:\n"
                "   - Hybrid RAG combines multiple distinct retrieval approaches—typically semantic/dense vector search (meaning/similarity) and keyword/sparse search (exact matches/BM25)—to retrieve relevant candidate chunks from complementary perspectives. Their results are then combined/fused (e.g., via Reciprocal Rank Fusion) and optionally reranked before being passed to the LLM.\n"
                "   - Clearly maintain the distinction that Hybrid RAG is broader than simply 'RAG + reranking.' Reranking can be a downstream component of any retrieval pipeline, but the combination of multiple retrieval signals/methods (semantic + keyword) is what fundamentally makes a system hybrid.\n"
                "3. SUMMARY & OVERVIEW INQUIRIES: If the user asks what the document is about, or asks for a summary, overview, or main topics, "
                "provide a thorough, well-structured, and helpful synthesis of the document using the Document Context and Excerpts.\n"
                "4. OUT-OF-CONTEXT / UNRELATED TOPICS: If the user's question asks about a person, entity, company, or topic that is completely outside the scope "
                "of the document (for example, asking about an unrelated celebrity, sports, or external topic not in the document), "
                "you MUST NOT answer the out-of-context query using general external knowledge. "
                "Instead, respond properly, politely, and informatively: clarify what topic the active document focuses on, explain that it does not contain information on the requested query, "
                "and invite the user to ask any questions related to the document.\n"
                "5. MANDATORY MARKDOWN FORMATTING: You MUST ALWAYS format your entire response using rich GitHub-flavored Markdown. Organize your response with descriptive headings (##, ###), bullet points, bold key terms, blockquotes, and syntax-highlighted code blocks where appropriate. NEVER output raw wall-of-text paragraphs without markdown formatting.\n"
                "6. INLINE MULTIMODAL IMAGES & DIAGRAMS: When context excerpts contain visual assets or image diagrams (indicated by Image URL: http...), if relevant to the user query, seamlessly render the markdown image tag '![Description](URL)' inline in your response to illustrate the concepts. "
                "If the user asks about diagrams, illustrations, charts, or visuals in the article, or asks to display them, you MUST enumerate the diagrams found in the excerpts and render their inline image tags '![Description](URL)' along with a clear summary of what each diagram shows."
                "6. INLINE MULTIMODAL IMAGES & DIAGRAMS: When context excerpts contain visual assets or image diagrams (indicated by 'Image URL: http...'), seamlessly render the markdown image tag '![Description](URL)' inline in your response to visually illustrate the concepts. "
                "If the user asks about diagrams, illustrations, charts, or visuals in the article, or asks to display them, you MUST enumerate all diagrams found in the excerpts, provide a concise explanation of what architectural concepts each diagram illustrates, and render its inline markdown image tag '![Description](URL)' using the EXACT URL specified in the excerpt. NEVER output placeholder text like '(source)' or omit image tags when visual assets are present in the excerpts."
            )

            formatted_context = ""
            for i, chunk in enumerate(context_chunks, 1):
                formatted_context += f"--- Excerpt [{i}] ---\n{chunk['content']}\n\n"

            memories_text = ""
            if user_memories:
                memories_text = "User Facts & Semantic Profile:\n" + "\n".join(f"- {m}" for m in user_memories) + "\n\n"

            history_text = ""
            if chat_history:
                recent = chat_history[-6:]  # recent conversation turns
                for msg in recent:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_text += f"{role}: {msg.get('content', '')}\n"

            prompt = (
                f"{doc_header}"
                f"Context Excerpts from Website / Document:\n{formatted_context}\n"
                f"{memories_text}"
                f"Conversation History:\n{history_text}\n"
                f"Current User Query: {query}\n\n"
                f"Answer:"
            )

            return prompt, system_instruction
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}", exc_info=True)
            return query, ""


rag_service = RAGService()


