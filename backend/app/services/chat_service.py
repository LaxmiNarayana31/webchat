import asyncio
import re
from typing import AsyncGenerator, List, Dict, Any, Optional
import uuid

from backend.app.agents.chat_agent import chat_agent
from backend.app.cache.semantic_cache import semantic_cache_service
from backend.app.cache.vector_cache import vector_store_cache
from backend.app.core.errors import (
    ScraperException,
    WebChatException,
)
from backend.app.core.logging import logger
from backend.app.dtos.chat_dto import ChatRequestDto, ChatResponseDto, CitationItemDto
from backend.app.dtos.model_dto import ModelCatalogItemDto
from backend.app.dtos.scrape_dto import (
    CrawlPageDto,
    CrawlRequestDto,
    CrawlResponseDto,
    ScrapeRequestDto,
    ScrapeResponseDto,
)
from backend.app.helpers.stream_helper import format_done_event, format_sse_event
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.repositories.session_repository import session_repository
from backend.app.repositories.url_cache_repository import url_cache_repository
from backend.app.services.llm_service import llm_service
from backend.app.services.memory_service import memory_service
from backend.app.services.rag_service import rag_service
from backend.app.services.scraper_service import scraper_service
from backend.app.services.user_service import user_service


class ChatService:
    """Orchestrates RAG workflows, streaming generation, ingestion, and catalog operations."""

    def handle_scrape_and_index(self, req: ScrapeRequestDto) -> ScrapeResponseDto:
        """Scrapes and indexes URL content (single or multiple) into vector store, checking cache first."""
        try:
            raw_urls = []
            if req.urls:
                raw_urls.extend(req.urls)
            elif req.url:
                parts = [p.strip() for p in re.split(r"[\r\n,]+", req.url) if p.strip()]
                raw_urls.extend(parts)

            urls = []
            for u in raw_urls:
                u_clean = u.strip()
                if u_clean and u_clean not in urls:
                    if not u_clean.startswith(("http://", "https://")):
                        u_clean = "https://" + u_clean
                    urls.append(u_clean)

            if not urls:
                raise ScraperException("Please provide at least one valid URL.")

            # --- Single URL Flow ---
            if len(urls) == 1:
                target_url = urls[0]
                url_hash = compute_url_hash(target_url)
                cached = url_cache_repository.get_by_hash(url_hash)
                meta_cached = (cached.get("metadata") or {}) if cached else {}
                if cached and "images" in meta_cached and meta_cached.get("images") is not None:
                    logger.info(f"ChatService: Cache HIT for URL '{target_url}' (hash: {url_hash[:12]}...)")
                    meta_cached = cached.get("metadata") or {}
                    if not vector_store_cache.get(url_hash) and cached.get("vector_session_id"):
                        vs_idx = rag_service.get_session_index(
                            cached["vector_session_id"],
                            metadata=meta_cached,
                        )
                        vector_store_cache.set(url_hash, vs_idx, metadata=meta_cached)
                        vector_store_cache.set(target_url, vs_idx, metadata=meta_cached)

                    return ScrapeResponseDto(
                        success=True,
                        url=cached.get("url") or target_url,
                        title=cached.get("title") or "Extracted Document",
                        content=cached.get("content") or "",
                        word_count=cached.get("word_count", 0),
                        strategy_used=cached.get("strategy_used", "cached"),
                        paywall_detected=False,
                        paywall_bypassed=cached.get("paywall_bypassed", False),
                        pages_crawled=meta_cached.get("pages_crawled"),
                    )

                # Check if this URL is a site root or documentation hub that should be deep-crawled automatically
                auto_crawl_target = req.auto_crawl and scraper_service.should_auto_crawl(target_url)

                if auto_crawl_target:
                    logger.info(f"ChatService: Auto-crawling site/documentation hub for '{target_url}' (strategy: {req.strategy})")
                    crawl_res = scraper_service.crawl_domain(target_url, max_pages=8, max_depth=2, strategy=req.strategy)

                    if crawl_res.get("success") and crawl_res.get("pages_crawled", 0) > 0:
                        content = crawl_res.get("content", "")
                        final_url = crawl_res.get("root_url", target_url)
                        title = crawl_res.get("title", f"Site Crawl: {final_url}")
                        total_words = crawl_res.get("total_words", 0)
                        pages_crawled = crawl_res.get("pages_crawled", 1)
                        crawled_subpages = crawl_res.get("pages", [])
                        vec_sid = str(uuid.uuid4())

                        meta = {
                            "url": final_url,
                            "title": title,
                            "session_id": vec_sid,
                            "vector_session_id": vec_sid,
                            "url_hash": url_hash,
                            "is_crawl": True,
                            "pages_crawled": pages_crawled,
                            "pages": crawled_subpages,
                        }

                        vs, err = rag_service.build_vectorstore(content, metadata=meta)
                        if vs:
                            vector_store_cache.set(url_hash, vector_store=vs, metadata=meta)
                            vector_store_cache.set(final_url, vector_store=vs, metadata=meta)
                        elif err:
                            logger.warning(f"ChatService: Auto-crawl vector indexing warning: {err}")

                        url_cache_repository.save_url_cache(
                            url_hash=url_hash,
                            url=final_url,
                            title=title,
                            vector_session_id=vec_sid,
                            word_count=total_words,
                            strategy_used="auto_crawl",
                            paywall_bypassed=False,
                            metadata=meta,
                            content=content[:100000],
                        )

                        return ScrapeResponseDto(
                            success=True,
                            url=final_url,
                            title=title,
                            content=content,
                            word_count=total_words,
                            strategy_used="auto_crawl",
                            paywall_detected=False,
                            paywall_bypassed=False,
                            pages_crawled=pages_crawled,
                        )
                    else:
                        logger.warning(f"ChatService: Auto-crawl failed or returned 0 pages, falling back to direct scrape: {crawl_res.get('error')}")

                logger.info(f"ChatService: Scraping single URL '{target_url}' (strategy: {req.strategy})")
                res = scraper_service.scrape_url(target_url, strategy=req.strategy)

                if not res.get("success"):
                    err_msg = res.get("error", "Unable to extract readable content from URL.")
                    logger.error(f"ChatService: Scrape failed for '{target_url}': {err_msg}")
                    raise ScraperException(message=err_msg, details={"url": target_url})

                content = res.get("content", "")
                final_url = res.get("url", target_url)
                title = res.get("title", "Extracted Document")
                vec_sid = str(uuid.uuid4())

                meta = {
                    "url": final_url,
                    "title": title,
                    "session_id": vec_sid,
                    "vector_session_id": vec_sid,
                    "url_hash": url_hash,
                    "images": res.get("images", []),
                }

                vs, err = rag_service.build_vectorstore(content, metadata=meta)
                if vs:
                    vector_store_cache.set(url_hash, vector_store=vs, metadata=meta)
                    vector_store_cache.set(final_url, vector_store=vs, metadata=meta)
                elif err:
                    logger.warning(f"ChatService: Vector pre-indexing warning: {err}")

                url_cache_repository.save_url_cache(
                    url_hash=url_hash,
                    url=final_url,
                    title=title,
                    vector_session_id=vec_sid,
                    word_count=res.get("word_count", 0),
                    strategy_used=res.get("strategy_used", ""),
                    paywall_bypassed=res.get("paywall_bypassed", False),
                    metadata=meta,
                    content=content[:100000],
                )

                return ScrapeResponseDto(
                    success=True,
                    url=final_url,
                    title=title,
                    content=content,
                    word_count=res.get("word_count", 0),
                    strategy_used=res.get("strategy_used", ""),
                    paywall_detected=res.get("paywall_detected", False),
                    paywall_bypassed=res.get("paywall_bypassed", False),
                    pages_crawled=1,
                )

            # --- Multiple URLs Flow ---
            logger.info(f"ChatService: Ingesting {len(urls)} URLs simultaneously (strategy: {req.strategy})")
            scraped_sources = []
            any_paywall_bypassed = False
            total_words = 0

            for idx, target_url in enumerate(urls):
                url_hash = compute_url_hash(target_url)
                cached = url_cache_repository.get_by_hash(url_hash)

                if cached and cached.get("content"):
                    logger.info(f"ChatService: Multi-URL cache HIT [{idx+1}/{len(urls)}] for '{target_url}'")
                    scraped_sources.append({
                        "url": cached.get("url") or target_url,
                        "title": cached.get("title") or f"Source {idx+1}",
                        "content": cached.get("content") or "",
                        "word_count": cached.get("word_count", 0),
                        "paywall_bypassed": cached.get("paywall_bypassed", False),
                    })
                    if cached.get("paywall_bypassed"):
                        any_paywall_bypassed = True
                    total_words += cached.get("word_count", 0)
                    continue

                logger.info(f"ChatService: Multi-URL scraping [{idx+1}/{len(urls)}] for '{target_url}'")
                res = scraper_service.scrape_url(target_url, strategy=req.strategy)

                if res.get("success"):
                    src_content = res.get("content", "")
                    src_title = res.get("title", f"Source {idx+1}")
                    src_url = res.get("url", target_url)
                    src_words = res.get("word_count", 0)
                    bypassed = res.get("paywall_bypassed", False)
                    vec_sid = str(uuid.uuid4())

                    if bypassed:
                        any_paywall_bypassed = True
                    total_words += src_words

                    scraped_sources.append({
                        "url": src_url,
                        "title": src_title,
                        "content": src_content,
                        "word_count": src_words,
                        "paywall_bypassed": bypassed,
                        "images": res.get("images", []),
                    })

                    # Cache each individual URL for future instant hits
                    url_cache_repository.save_url_cache(
                        url_hash=url_hash,
                        url=src_url,
                        title=src_title,
                        vector_session_id=vec_sid,
                        word_count=src_words,
                        strategy_used=res.get("strategy_used", ""),
                        paywall_bypassed=bypassed,
                        metadata={"url": src_url, "title": src_title, "session_id": vec_sid, "images": res.get("images", [])},
                        content=src_content[:100000],
                    )
                else:
                    logger.warning(f"ChatService: Multi-URL scrape warning for '{target_url}': {res.get('error')}")

            if not scraped_sources:
                raise ScraperException("Unable to extract content from any of the provided URLs.")

            # Combine sources into a unified document
            combined_blocks = []
            source_titles = []
            all_images = []
            for i, src in enumerate(scraped_sources):
                source_titles.append(src["title"])
                all_images.extend(src.get("images", []))
                combined_blocks.append(
                    f"### Source {i+1}: {src['title']}\n**URL:** {src['url']}\n\n{src['content']}\n"
                )

            composite_content = "\n\n---\n\n".join(combined_blocks)
            composite_title = f"{len(scraped_sources)} Sources: {source_titles[0]}" + (f" and {len(scraped_sources)-1} more" if len(scraped_sources) > 1 else "")
            composite_url = ", ".join([s["url"] for s in scraped_sources])
            composite_hash = compute_url_hash(composite_url)
            composite_vec_sid = str(uuid.uuid4())

            composite_meta = {
                "url": composite_url,
                "title": composite_title,
                "session_id": composite_vec_sid,
                "is_multi_source": True,
                "sources_count": len(scraped_sources),
                "images": all_images[:20],
            }

            vs, err = rag_service.build_vectorstore(composite_content, metadata=composite_meta)
            if vs:
                vector_store_cache.set(composite_hash, vector_store=vs, metadata=composite_meta)
                vector_store_cache.set(composite_url, vector_store=vs, metadata=composite_meta)
            elif err:
                logger.warning(f"ChatService: Multi-URL vector indexing warning: {err}")

            return ScrapeResponseDto(
                success=True,
                url=composite_url,
                title=composite_title,
                content=composite_content,
                word_count=total_words,
                strategy_used=req.strategy,
                paywall_detected=False,
                paywall_bypassed=any_paywall_bypassed,
            )
        except WebChatException:
            raise
        except Exception as e:
            logger.error(f"ChatService: Unexpected error in scrape_and_index: {e}", exc_info=True)
            raise WebChatException(message=f"Scraping failed: {str(e)}", status_code=500)

    def handle_crawl_and_index(self, req: CrawlRequestDto) -> CrawlResponseDto:
        """Recursively crawls linked pages within the same domain and indexes into vector cache."""
        try:
            logger.info(f"ChatService: Deep crawling root '{req.url}' (max_pages={req.max_pages}, max_depth={req.max_depth})")
            crawl_res = scraper_service.crawl_domain(
                start_url=req.url,
                max_pages=req.max_pages,
                max_depth=req.max_depth,
                strategy=req.strategy,
            )

            if not crawl_res.get("success"):
                err_msg = crawl_res.get("error", f"Failed to crawl domain from {req.url}")
                logger.error(f"ChatService: Domain crawl failed for '{req.url}': {err_msg}")
                raise ScraperException(message=err_msg, details={"url": req.url})

            content = crawl_res.get("content", "")
            root_url = crawl_res.get("root_url", req.url)
            title = crawl_res.get("title", f"Crawl: {root_url}")

            # Pre-build hierarchical vector store
            vs, err = rag_service.build_vectorstore(
                content, metadata={"url": root_url, "title": title, "is_crawl": True}
            )
            if vs:
                vector_store_cache.set(root_url, vector_store=vs, metadata=crawl_res)
            elif err:
                logger.warning(f"ChatService: Vector crawl indexing warning: {err}")

            page_dtos = [
                CrawlPageDto(
                    url=p.get("url", ""),
                    title=p.get("title", ""),
                    word_count=p.get("word_count", 0),
                    strategy_used=p.get("strategy_used", ""),
                )
                for p in crawl_res.get("pages", [])
            ]

            return CrawlResponseDto(
                success=True,
                root_url=root_url,
                pages_crawled=crawl_res.get("pages_crawled", 0),
                total_words=crawl_res.get("total_words", 0),
                pages=page_dtos,
            )
        except WebChatException:
            raise
        except Exception as e:
            logger.error(f"ChatService: Unexpected error in crawl_and_index: {e}", exc_info=True)
            raise WebChatException(message=f"Domain crawl failed: {str(e)}", status_code=500)

    # def handle_file_upload_and_index(self, file_bytes: bytes, filename: str) -> FileUploadResponseDto:
    #     """Parses uploaded file, extracts text, and indexes into vector cache."""
    #     try:
    #         logger.info(f"ChatService: Processing uploaded file '{filename}' ({len(file_bytes)} bytes)")
    #         doc_res = scraper_service.extract_from_file(file_bytes, filename)
    #
    #         if not doc_res.get("success"):
    #             err_msg = doc_res.get("error", f"Could not read text from '{filename}'.")
    #             raise ScraperException(message=err_msg, details={"filename": filename})
    #
    #         content = doc_res.get("content", "")
    #         title = doc_res.get("title", filename)
    #         virtual_url = f"file://{filename}"
    #
    #         meta = {
    #             "url": virtual_url,
    #             "title": title,
    #             "filename": filename,
    #             "file_type": doc_res.get("file_type", "FILE"),
    #             "word_count": doc_res.get("word_count", 0),
    #         }
    #
    #         # Build hierarchical vector store
    #         vs, err = rag_service.build_vectorstore(content, metadata=meta)
    #         if vs:
    #             vector_store_cache.set(virtual_url, vector_store=vs, metadata=meta)
    #         elif err:
    #             logger.warning(f"ChatService: Vector file indexing warning: {err}")
    #
    #         return FileUploadResponseDto(
    #             success=True,
    #             filename=filename,
    #             file_type=doc_res.get("file_type", "FILE"),
    #             title=title,
    #             word_count=doc_res.get("word_count", 0),
    #             virtual_url=virtual_url,
    #         )
    #     except WebChatException:
    #         raise
    #     except Exception as e:
    #         logger.error(f"ChatService: File upload indexing failed: {e}", exc_info=True)
    #         raise WebChatException(message=f"File processing failed: {str(e)}", status_code=500)

    async def handle_chat_query_async(self, req: ChatRequestDto, client_ip: Optional[str] = None) -> ChatResponseDto:
        """Executes non-streaming conversational RAG, checking dual-layer quota and persisting conversation turn asynchronously."""
        try:
            # Quota Check (Dual-layer: Client ID + Client IP)
            allowed, err_msg, quota_info = user_service.check_and_consume_quota(
                email=None,
                client_id=req.client_id,
                ip_address=client_ip,
            )
            if not allowed:
                raise WebChatException(
                    message=err_msg,
                    status_code=429,
                    details={"quota": quota_info, "requires_email": quota_info.get("requires_email", False)},
                )

            session_id = req.session_id or str(uuid.uuid4())
            user_identifier = req.user_email or req.client_id or "anonymous"
            memory_service.set_attribution(user_identifier)

            # Persist user question in DB
            session_repository.append_message(
                session_id=session_id,
                role="user",
                content=req.query,
                url=req.url,
                email=None,
                guest_client_id=req.client_id,
            )

            # Check semantic cache first
            cached = semantic_cache_service.get_cached_answer(query=req.query, context_hash=req.url)
            if cached and "Content Unavailable" not in cached.get("answer", ""):
                raw_cits = cached.get("citations", [])
                safe_citations = []
                for idx, c in enumerate(raw_cits):
                    if isinstance(c, dict):
                        safe_citations.append(
                            CitationItemDto(
                                chunk_index=c.get("chunk_index", idx),
                                content=c.get("content") or c.get("text", ""),
                                title=c.get("title", ""),
                                url=c.get("url", ""),
                            )
                        )
                    elif hasattr(c, "chunk_index"):
                        safe_citations.append(c)

                # Persist cached answer in DB
                session_repository.append_message(
                    session_id=session_id,
                    role="assistant",
                    content=cached["answer"],
                    model_used="cached",
                    provider="cache",
                    fallback_triggered=False,
                    latency_sec=0.01,
                    citations=[c.model_dump() if hasattr(c, "model_dump") else c for c in safe_citations],
                    url=req.url,
                    email=req.user_email,
                    guest_client_id=req.client_id,
                )
                return ChatResponseDto(
                    answer=cached["answer"],
                    model_used="cached",
                    provider="cache",
                    fallback_triggered=False,
                    latency_sec=0.01,
                    citations=safe_citations,
                    session_id=session_id,
                    quota=quota_info
                )

            # Generate grounded answer
            response_dto = await chat_agent.answer_query_async(
                query=req.query,
                url=req.url,
                document_content=req.document_content,
                chat_history=req.chat_history,
                selected_model=req.selected_model,
                user_id=user_identifier,
            )

            citations_list = [c.model_dump() for c in response_dto.citations]

            # Persist assistant answer in DB
            session_repository.append_message(
                session_id=session_id,
                role="assistant",
                content=response_dto.answer,
                model_used=response_dto.model_used,
                provider=response_dto.provider,
                fallback_triggered=response_dto.fallback_triggered,
                latency_sec=response_dto.latency_sec,
                citations=citations_list,
                url=req.url,
                email=req.user_email,
                guest_client_id=req.client_id,
            )

            # Extract and update semantic user memory
            try:
                memory_service.add_memory(
                    messages=f"User: {req.query}\nAssistant: {response_dto.answer}",
                    user_id=user_identifier,
                    session_id=session_id,
                )
            except Exception as mem_err:
                logger.warning(f"ChatService: Memory update warning: {mem_err}")

            response_dto.session_id = session_id
            response_dto.quota = quota_info

            # Save generated answer to semantic cache
            semantic_cache_service.set_cached_answer(
                query=req.query,
                answer=response_dto.answer,
                citations=citations_list,
                context_hash=req.url,
            )

            return response_dto

        except WebChatException:
            raise
        except Exception as e:
            logger.error(f"ChatService: Chat generation failed: {e}", exc_info=True)
            raise WebChatException(message=f"Chat generation failed: {str(e)}", status_code=500)

    async def handle_chat_stream_async(self, req: ChatRequestDto, client_ip: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Executes streaming conversational RAG via SSE, tracking dual-layer quota and persisting conversation turn asynchronously."""
        try:
            # Quota Check (Dual-layer: Client ID + Client IP)
            allowed, err_msg, quota_info = user_service.check_and_consume_quota(
                email=None,
                client_id=req.client_id,
                ip_address=client_ip,
            )
            if not allowed:
                raise WebChatException(
                    message=err_msg,
                    status_code=429,
                    details={"quota": quota_info, "requires_email": quota_info.get("requires_email", False)},
                )

            session_id = req.session_id or str(uuid.uuid4())
            user_identifier = req.user_email or req.client_id or "anonymous"
            memory_service.set_attribution(user_identifier)

            # Persist user message to DB
            session_repository.append_message(
                session_id=session_id,
                role="user",
                content=req.query,
                url=req.url,
                email=None,
                guest_client_id=req.client_id,
            )

            # Check semantic cache
            cached = semantic_cache_service.get_cached_answer(query=req.query, context_hash=req.url)
            if cached and "Content Unavailable" not in cached.get("answer", ""):
                sim_score = cached.get("similarity", 1.0)
                tier_name = cached.get("tier", "semantic")
                detail_text = (
                    "Instant sub-millisecond response (Tier 1 exact match)"
                    if tier_name == "exact"
                    else f"Sub-second response via Upstash Redis semantic vector match (similarity: {sim_score:.2f})"
                )
                yield format_sse_event("session", {"session_id": session_id, "quota": quota_info})
                yield format_sse_event("step", {
                    "step": "cache",
                    "icon": "⚡",
                    "title": f"Semantic Cache Hit ({tier_name.capitalize()})",
                    "detail": detail_text,
                    "status": "done"
                })
                if cached.get("citations"):
                    yield format_sse_event("citations", cached["citations"])
                
                # Stream the cached answer in small chunks to simulate generation
                words = cached["answer"].split(" ")
                for i in range(0, len(words), 3):
                    chunk_text = " ".join(words[i:i+3]) + " "
                    yield format_sse_event("message", {
                        "chunk": chunk_text,
                        "model_used": "cached",
                        "provider": "cache",
                        "fallback_triggered": False,
                        "latency_sec": 0.01,
                        "session_id": session_id,
                        "quota": quota_info
                    })
                    await asyncio.sleep(0.01)

                session_repository.append_message(
                    session_id=session_id,
                    role="assistant",
                    content=cached["answer"],
                    model_used="cached",
                    provider="cache",
                    fallback_triggered=False,
                    latency_sec=0.01,
                    citations=cached.get("citations", []),
                    url=req.url,
                    email=req.user_email,
                    guest_client_id=req.client_id,
                )
                yield format_done_event()
                return

            # Initialize Agent Stream
            events_gen = chat_agent.answer_query_stream_events_async(
                query=req.query,
                url=req.url,
                document_content=req.document_content,
                chat_history=req.chat_history,
                selected_model=req.selected_model,
                user_id=user_identifier,
            )

            # Stream session, reasoning steps, citations & tokens
            yield format_sse_event("session", {"session_id": session_id, "quota": quota_info})

            accumulated_answer: List[str] = []
            citations_data: List[Dict[str, Any]] = []
            final_model = "unknown"
            final_provider = "unknown"
            final_fallback = False
            final_latency = 0.0

            async for item in events_gen:
                item_type = item.get("type")
                if item_type == "step":
                    yield format_sse_event("step", item)
                elif item_type == "citations":
                    citations_data = item.get("citations", [])
                    yield format_sse_event("citations", citations_data)
                elif item_type == "chunk":
                    chunk_text = item.get("chunk", "")
                    if chunk_text:
                        accumulated_answer.append(chunk_text)
                    if item.get("model_used"):
                        final_model = item.get("model_used")
                    if item.get("provider"):
                        final_provider = item.get("provider")
                    if item.get("fallback_triggered"):
                        final_fallback = True
                    if item.get("latency_sec"):
                        final_latency = item.get("latency_sec")
                    item["session_id"] = session_id
                    item["quota"] = quota_info

                    yield format_sse_event("message", item)

            # Persist completed response to database
            full_text = "".join(accumulated_answer)
            session_repository.append_message(
                session_id=session_id,
                role="assistant",
                content=full_text,
                model_used=final_model,
                provider=final_provider,
                fallback_triggered=final_fallback,
                latency_sec=final_latency,
                citations=citations_data,
                url=req.url,
                email=req.user_email,
                guest_client_id=req.client_id,
            )

            # Record semantic memory
            try:
                memory_service.add_memory(
                    messages=f"User: {req.query}\nAssistant: {full_text}",
                    user_id=user_identifier,
                    session_id=session_id,
                )
            except Exception as mem_err:
                logger.warning(f"ChatService: Memory update warning in stream: {mem_err}")

            # Save generated answer to semantic cache
            semantic_cache_service.set_cached_answer(
                query=req.query,
                answer=full_text,
                citations=citations_data,
                context_hash=req.url,
            )

            # Terminal event
            yield format_done_event()

        except WebChatException as wce:
            logger.warning(f"ChatService: Stream handled exception: {wce.message}")
            yield format_sse_event("error", {"message": wce.message, "status_code": wce.status_code})
            yield format_done_event()
        except Exception as e:
            logger.error(f"ChatService: Streaming failed: {e}", exc_info=True)
            yield format_sse_event("error", {"message": str(e), "status_code": 500})
            yield format_done_event()

    def get_model_catalog(self) -> List[ModelCatalogItemDto]:
        """Returns catalog of models with live rate limiter telemetry."""
        try:
            return llm_service.get_model_catalog()  # type: ignore
        except Exception as e:
            logger.error(f"ChatService: Failed to retrieve model catalog: {e}")
            return []


chat_service = ChatService()
