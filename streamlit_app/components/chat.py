import re
import time
import uuid

import streamlit as st

from backend.app.agents.chat_agent import chat_agent
from backend.app.cache.vector_cache import vector_store_cache
from backend.app.core.logging import logger
from backend.app.dtos.chat_dto import ChatMessageDto
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.repositories.session_repository import session_repository
from backend.app.repositories.url_cache_repository import url_cache_repository
from backend.app.services.rag_service import rag_service
from backend.app.services.scraper_service import scraper_service
from backend.app.services.user_service import user_service


def typewriter_effect(text: str, speed: int = 35):
    """Renders text with a smooth typewriter cadence."""
    try:
        container = st.empty()
        accumulated = ""
        words = text.split(" ")
        for i, word in enumerate(words):
            accumulated += word + " "
            if i % 2 == 0 or i == len(words) - 1:
                container.markdown(accumulated + "▌")
                time.sleep(1.0 / max(1, speed))
        container.markdown(accumulated.strip())
    except Exception as e:
        logger.error(f"Error in typewriter_effect: {e}", exc_info=True)
        st.markdown(text)


def trigger_quick_ingest(raw_url_input: str, strategy: str = "auto"):
    """Helper to ingest single or multiple website URLs, index visual diagrams, and initialize chat state."""
    try:
        raw_urls = [u.strip() for u in re.split(r"[\n,]+", raw_url_input) if u.strip()]
        if not raw_urls:
            st.warning("Please enter at least one valid website URL.")
            return

        # Clean URL protocols
        cleaned_urls = []
        for u in raw_urls:
            if not u.lower().startswith("http://") and not u.lower().startswith("https://"):
                u = f"https://{u}"
            cleaned_urls.append(u)

        url_hash = compute_url_hash(cleaned_urls)
        combined_urls_str = ", ".join(cleaned_urls)

        # Check database URL cache first: Avoid expensive re-scraping and embedding generation
        cached = url_cache_repository.get_by_hash(url_hash)
        if cached:
            primary_title = cached.get("title") or "Website Knowledge Base"
            total_words = cached.get("word_count", 0)
            any_paywall_bypassed = cached.get("paywall_bypassed", False)
            cached_meta = cached.get("metadata") or {}
            vec_sid = cached.get("vector_session_id")

            # Retrieve existing vector store from memory/disk cache or Qdrant session index
            vs = vector_store_cache.get(url_hash)
            if not vs and vec_sid:
                vs = rag_service.get_session_index(
                    session_id=vec_sid,
                    metadata=cached_meta,
                )
            elif vs and vec_sid:
                vs.session_id = vec_sid

            # Privacy & isolation: Generate brand new unique session_id
            new_sid = str(uuid.uuid4())
            st.session_state.session_id = new_sid
            st.query_params["sid"] = new_sid
            st.session_state.selected_chat_index = None

            meta = {
                **cached_meta,
                "url": cached.get("url") or combined_urls_str,
                "title": primary_title,
                "word_count": total_words,
                "strategy_used": cached.get("strategy_used", "cached"),
                "paywall_detected": False,
                "paywall_bypassed": any_paywall_bypassed,
                "urls": cleaned_urls,
                "session_id": new_sid,
                "vector_session_id": vec_sid,
                "cached_hit": True,
                "url_hash": url_hash,
            }

            st.session_state.vector_store = vs
            st.session_state.site_metadata = meta
            st.session_state.current_chat = []  # Clean empty chat state ready for questions

            st.success(f"Instant vector cache hit. Knowledge base ready for **{primary_title}**.")
            st.rerun()

        # Cache miss: Scrape, vectorize, and persist to database URL cache
        with st.spinner(f"Extracting & indexing {len(cleaned_urls)} website{'s' if len(cleaned_urls) > 1 else ''} with diagrams..."):
            st.session_state.selected_chat_index = None
            extracted_sections = []
            titles = []
            total_words = 0
            strategies_used = set()
            any_paywall_bypassed = False
            valid_urls = []

            for url in cleaned_urls:
                scrape_res = scraper_service.scrape_url(url, strategy=strategy)
                if scrape_res.get("success"):
                    content = scrape_res.get("content", "")
                    title = scrape_res.get("title", url)
                    word_count = scrape_res.get("word_count", 0)

                    extracted_sections.append(f"# Website: {title}\nURL: {url}\n\n{content}")
                    titles.append(title)
                    total_words += word_count
                    strategies_used.add(scrape_res.get("strategy_used", "auto"))
                    if scrape_res.get("paywall_bypassed"):
                        any_paywall_bypassed = True
                    valid_urls.append(url)
                else:
                    st.warning(f"Could not extract `{url}`: {scrape_res.get('error', 'Extraction failed')}")

            if not extracted_sections:
                st.error("Unable to extract content from any of the specified URLs.")
                return

            combined_text = "\n\n" + ("=" * 40) + "\n\n" + ("\n\n" + ("=" * 40) + "\n\n").join(extracted_sections)
            primary_title = titles[0] if len(titles) == 1 else f"{titles[0]} (+{len(titles) - 1} more)"
            combined_urls_str = ", ".join(valid_urls)

            new_sid = str(uuid.uuid4())
            st.session_state.session_id = new_sid
            st.query_params["sid"] = new_sid

            meta = {
                "url": combined_urls_str,
                "title": primary_title,
                "word_count": total_words,
                "strategy_used": ", ".join(strategies_used),
                "paywall_detected": False,
                "paywall_bypassed": any_paywall_bypassed,
                "urls": valid_urls,
                "session_id": new_sid,
                "vector_session_id": new_sid,
                "cached_hit": False,
                "url_hash": url_hash,
            }

            vs, err = rag_service.build_vectorstore(combined_text, metadata=meta)
            if err:
                st.error(f"Vector indexing failed: {err}")
                return

            st.session_state.vector_store = vs
            st.session_state.site_metadata = meta
            vector_store_cache.set(url_hash, vs, metadata=meta)

            # Store in database URL cache for future sessions
            url_cache_repository.save_url_cache(
                url_hash=url_hash,
                url=combined_urls_str,
                title=primary_title,
                vector_session_id=new_sid,
                word_count=total_words,
                strategy_used=", ".join(strategies_used),
                paywall_bypassed=any_paywall_bypassed,
                metadata=meta,
                content=combined_text[:100000],
            )

            st.session_state.current_chat = []  # Clean empty chat state ready for questions
            st.success(f"Successfully indexed {len(valid_urls)} website{'s' if len(valid_urls) > 1 else ''} with multimodal diagrams!")
            st.rerun()
    except Exception as e:
        logger.error(f"Error in trigger_quick_ingest: {e}", exc_info=True)
        st.error(f"Ingestion failed: {e}")


def render_chat():
    """Renders the conversational chat interface, suggested inquiries, and streaming answers."""
    try:
        messages = st.session_state.get("current_chat", [])
        has_site = bool(st.session_state.get("site_metadata"))

        # --- View 1: Landing Page (No site loaded and no messages) ---
        if not messages and not has_site:
            st.markdown(
                """<div class="hero-landing-wrapper">
    <div class="hero-super-badge">✨ Autonomous Web Intelligence & Agentic RAG</div>
    <h1 class="hero-heading-gradient">Chat with Any Web Source</h1>
    <p class="hero-subtitle-text">Extract articles, bypass paywalls, crawl documentation, and ground your questions with multi-model validation.</p>
</div>""",
                unsafe_allow_html=True,
            )

            st.markdown(
                """<div class="ingestion-card">
    <div class="ingestion-card-header">
        <div class="ingestion-card-title">🌐 Target Website URL(s)</div>
        <span style="font-size: 11px; padding: 2px 8px; border-radius: 9999px; background: rgba(255,255,255,0.06); color: #94a3b8; font-weight: 500;">Multi-URL Supported</span>
    </div>
</div>""",
                unsafe_allow_html=True,
            )

            with st.form("quick_ingest_form", border=False):
                raw_url_input = st.text_area(
                    "Website URL",
                    placeholder="Paste website URL(s) here (e.g. example.com, https://docs.python.org)...",
                    height=90,
                    label_visibility="collapsed",
                    key="main_quick_url_input",
                )

                col_info, col_btn = st.columns([5.2, 2.3], gap="small", vertical_alignment="center")
                with col_info:
                    st.markdown(
                        "<div style='font-size: 11.5px; color: #64748b; line-height: 1.4; padding-left: 2px;'>Automatic paywall bypass and multi-provider vector indexing.</div>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    ingest_clicked = st.form_submit_button("⚡ Ingest & Start Chat", use_container_width=True)

                if ingest_clicked:
                    if raw_url_input and raw_url_input.strip():
                        trigger_quick_ingest(raw_url_input.strip())
                    else:
                        st.warning("Please enter at least one valid website URL.")

            # Feature capabilities grid matching React frontend
            st.markdown(
                """<div class="capabilities-grid">
    <div class="cap-card">
        <div class="cap-icon" style="font-size: 20px;">📚</div>
        <div class="cap-title">Agentic CRAG</div>
        <div class="cap-desc">Query rewriting, web fallback, and groundedness checks with LangGraph.</div>
    </div>
    <div class="cap-card">
        <div class="cap-icon" style="font-size: 20px;">🛡️</div>
        <div class="cap-title">Paywall Bypass</div>
        <div class="cap-desc">Apollo State extraction, Jina headless reading, and Wayback Machine fallback.</div>
    </div>
    <div class="cap-card">
        <div class="cap-icon" style="font-size: 20px;">⚡</div>
        <div class="cap-title">10-Tier Failover</div>
        <div class="cap-desc">Gemini 3.6 Flash auto-cascading to Groq LLaMA models with circuit breakers.</div>
    </div>
</div>""",
                unsafe_allow_html=True,
            )

            return

        # --- View 2: Active Document Banner with Quick Controls ---
        if has_site:
            meta = st.session_state.site_metadata
            title_str = meta.get("title", "Website Content")

            col_banner_info, col_banner_actions = st.columns([7.0, 3.0], vertical_alignment="center")
            with col_banner_info:
                badge_text = "⚡ Cached" if meta.get("cached_hit") else "✨ Newly Indexed"
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 10px; padding: 8px 14px; background: rgba(18, 22, 36, 0.85); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px;">
                        <span style="font-size: 16px;">🌐</span>
                        <div style="min-width: 0; overflow: hidden; flex: 1;">
                            <div style="font-size: 13px; font-weight: 600; color: #f8fafc; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                                {title_str}
                                <span style="font-size: 10px; margin-left: 6px; padding: 2px 7px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-radius: 9999px; font-weight: 500;">{badge_text}</span>
                            </div>
                            <div style="font-size: 11px; color: #64748b; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                                {meta.get('url', '')}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_banner_actions:
                col_btn_nc, col_btn_cu = st.columns([1, 1], gap="small")
                with col_btn_nc:
                    if st.button("🔄 New Chat", key="banner_btn_nc", use_container_width=True, help="Start a new chat session on this document"):
                        new_sid = str(uuid.uuid4())
                        st.session_state.session_id = new_sid
                        st.query_params["sid"] = new_sid
                        st.session_state.current_chat = []
                        st.session_state.selected_chat_index = None
                        st.rerun()
                with col_btn_cu:
                    if st.button("✕ Change URL", key="banner_btn_cu", use_container_width=True, help="Return to home screen to ingest a new website"):
                        st.query_params["sid"] = "new"
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.current_chat = []
                        st.session_state.vector_store = None
                        st.session_state.site_metadata = None
                        st.session_state.selected_chat_index = None
                        st.session_state.main_quick_url_input = ""
                        st.rerun()

            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # --- View 3: Knowledge Base Ready with Suggested Inquiry Chips (when current_chat is empty) ---
        if not messages and has_site:
            meta = st.session_state.site_metadata
            doc_title = meta.get("title", meta.get("url", "this document"))

            st.markdown(
                f"""
                <div class="kb-ready-box">
                    <div style="font-size: 28px; margin-bottom: 6px;">✨</div>
                    <div class="kb-ready-title">Knowledge Base Ready</div>
                    <div class="kb-ready-desc">
                        Select a suggested inquiry below or ask any question about <strong>{doc_title}</strong>:
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            suggested_questions = [
                "📌 What are the key takeaways and summary of this document?",
                "🔍 Explain the core concepts and architecture step-by-step",
                "📊 What architectural diagrams or illustrations are included? Display them",
                "⚡ What are the main performance trade-offs and limitations mentioned?",
            ]

            col1, col2 = st.columns([1, 1], gap="small")
            for idx, q_text in enumerate(suggested_questions):
                target_col = col1 if idx % 2 == 0 else col2
                with target_col:
                    clean_query = re.sub(r"^[📌🔍📊⚡]\s*", "", q_text).strip()
                    if st.button(q_text, key=f"btn_chip_{idx}", use_container_width=True):
                        st.session_state.pending_query = clean_query
                        st.rerun()

        # --- View 4: Conversation Turn Stream ---
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            steps = msg.get("steps", [])

            with st.chat_message(role):
                if role == "assistant" and steps:
                    with st.expander(f"Thought Process ({len(steps)} steps)", expanded=False):
                        for s in steps:
                            icon = s.get("icon", "•")
                            title = s.get("title", "Step")
                            detail = s.get("detail", "")
                            st.markdown(f"{icon} **{title}**: `{detail}`")

                st.markdown(content)

        # --- User Prompt Input (via Chat Input or Suggested Inquiry Button) ---
        placeholder_text = (
            f"Ask a question about \"{st.session_state.site_metadata.get('title', 'this document')[:35]}...\""
            if st.session_state.get("site_metadata")
            else "Ask anything..."
        )
        typed_query = st.chat_input(placeholder_text)

        active_query = None
        if typed_query and typed_query.strip():
            active_query = typed_query.strip()
        elif st.session_state.get("pending_query"):
            active_query = st.session_state.pop("pending_query")

        if active_query:
            client_id = st.session_state.get("client_id")
            user_email = (st.session_state.get("user_email") or "").strip().lower() or None
            allowed, err_msg, quota_info = user_service.check_and_consume_quota(email=user_email, client_id=client_id)
            allowed, err_msg, quota_info = user_service.check_and_consume_quota(
                email=user_email,
                client_id=client_id,
                ip_address="127.0.0.1",
            )

            if not allowed:
                st.error(f"⚠️ {err_msg}")
                return

            session_id = st.session_state.get("session_id") or str(uuid.uuid4())
            st.session_state.session_id = session_id
            st.query_params["sid"] = session_id

            # Save user turn in DB
            session_repository.append_message(
                session_id=session_id,
                role="user",
                content=active_query,
                url=st.session_state.site_metadata.get("url") if st.session_state.get("site_metadata") else None,
                email=user_email,
                guest_client_id=client_id,
            )

            # Append to current chat state
            st.session_state.current_chat.append({"role": "user", "content": active_query})

            with st.chat_message("user"):
                st.markdown(active_query)

            if not st.session_state.get("vector_store"):
                with st.chat_message("assistant"):
                    st.warning("Please ingest a website first before asking questions.")
            else:
                with st.chat_message("assistant"):
                    try:
                        history_dtos = [
                            ChatMessageDto(role=m.get("role", "user"), content=m.get("content", ""))
                            for m in st.session_state.current_chat[:-1]
                        ]

                        recorded_steps = []
                        citations_data = []
                        stream_meta = {"model_used": "auto", "provider": "ai", "fallback_triggered": False, "latency_sec": 0.0}

                        with st.status("Agent Reasoning & Retrieving Knowledge...", expanded=True) as status_box:
                            events_gen = chat_agent.answer_query_stream_events(
                                query=active_query,
                                url=st.session_state.site_metadata.get("url") if st.session_state.get("site_metadata") else None,
                                chat_history=history_dtos,
                                selected_model=st.session_state.get("selected_model_override"),
                                user_id=user_email or client_id,
                                vector_store=st.session_state.get("vector_store"),
                                document_metadata=st.session_state.get("site_metadata"),
                            )

                            def token_generator():
                                try:
                                    status_collapsed = False
                                    for item in events_gen:
                                        item_type = item.get("type")
                                        if item_type == "step":
                                            recorded_steps.append(item)
                                            icon = item.get("icon", "•")
                                            title = item.get("title", "Step")
                                            detail = item.get("detail", "")
                                            status_box.write(f"{icon} **{title}**: `{detail}`")
                                        elif item_type == "citations":
                                            citations_data.extend(item.get("citations", []))
                                        elif item_type == "chunk":
                                            if not status_collapsed:
                                                status_box.update(
                                                    label=f"Thought Process Complete ({len(recorded_steps)} steps)",
                                                    state="complete",
                                                    expanded=False,
                                                )
                                                status_collapsed = True

                                            if item.get("chunk"):
                                                yield item["chunk"]
                                            if item.get("done"):
                                                if item.get("model_used"):
                                                    stream_meta["model_used"] = item.get("model_used")
                                                    stream_meta["provider"] = item.get("provider", "ai")
                                                    stream_meta["fallback_triggered"] = item.get("fallback_triggered", False)
                                                    stream_meta["latency_sec"] = item.get("latency_sec", 0.0)
                                except Exception as token_err:
                                    logger.error(f"Error in token_generator: {token_err}", exc_info=True)
                                    yield f"\n\n[Generation error: {token_err}]"

                            answer = st.write_stream(token_generator())

                        # Persist assistant turn to DB
                        session_repository.append_message(
                            session_id=session_id,
                            role="assistant",
                            content=answer or "",
                            model_used=stream_meta.get("model_used", "auto"),
                            provider=stream_meta.get("provider", "ai"),
                            fallback_triggered=stream_meta.get("fallback_triggered", False),
                            latency_sec=stream_meta.get("latency_sec", 0.0),
                            citations=citations_data,
                            url=st.session_state.site_metadata.get("url") if st.session_state.get("site_metadata") else None,
                            email=user_email,
                            guest_client_id=client_id,
                        )

                        assistant_msg = {
                            "role": "assistant",
                            "content": answer,
                            "model_used": stream_meta.get("model_used", "auto"),
                            "provider": stream_meta.get("provider", "ai"),
                            "fallback_triggered": stream_meta.get("fallback_triggered", False),
                            "citations": citations_data,
                            "steps": recorded_steps,
                        }
                        st.session_state.current_chat.append(assistant_msg)
                        st.query_params["sid"] = session_id

                        if st.session_state.get("selected_chat_index") is not None:
                            idx = st.session_state.selected_chat_index
                            st.session_state.chat_histories[idx]["messages"] = st.session_state.current_chat

                        st.rerun()

                    except Exception as e:
                        logger.error(f"Error generating answer: {e}", exc_info=True)
                        st.error(f"Error generating answer: {e}")

    except Exception as e:
        logger.error(f"Error in render_chat: {e}", exc_info=True)
        st.error(f"Chat UI error: {e}")
