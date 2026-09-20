import uuid
import streamlit as st

from backend.app.cache.vector_cache import vector_store_cache
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.repositories.session_repository import session_repository
from backend.app.repositories.url_cache_repository import url_cache_repository
from backend.app.services.rag_service import rag_service
from backend.app.services.user_service import user_service


def render_sidebar():
    """Renders the sidebar: Document-aware New Chat, Session Search, Persistent History, and Quota Card."""
    try:
        with st.sidebar:
            client_id = st.session_state.get("client_id")
            user_email = (st.session_state.get("user_email") or "").strip().lower() or None
            has_doc = bool(st.session_state.get("site_metadata"))

            # --- Approach 1: Dual Action Buttons when document is active ---
            if has_doc:
                st.markdown(
                    """
                    <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                        Active Document Loaded
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "New Chat Session",
                    key="btn_sidebar_new_chat_doc",
                    use_container_width=True,
                    type="primary",
                    help="Start a fresh conversation while keeping current document indexed",
                ):
                    new_sid = str(uuid.uuid4())
                    st.session_state.session_id = new_sid
                    st.query_params["sid"] = new_sid
                    st.session_state.current_chat = []
                    st.session_state.selected_chat_index = None
                    st.rerun()

                if st.button(
                    "Ingest New URL",
                    key="btn_sidebar_ingest_new",
                    use_container_width=True,
                    help="Clear document and return to URL ingestion landing page",
                ):
                    new_sid = str(uuid.uuid4())
                    st.session_state.session_id = new_sid
                    if "sid" in st.query_params:
                        del st.query_params["sid"]
                    st.session_state.current_chat = []
                    st.session_state.vector_store = None
                    st.session_state.site_metadata = None
                    st.session_state.selected_chat_index = None
                    st.session_state.main_quick_url_input = ""
                    st.rerun()
            else:
                if st.button(
                    "Start New Session",
                    key="btn_sidebar_new_session",
                    use_container_width=True,
                    type="primary",
                ):
                    new_sid = str(uuid.uuid4())
                    st.session_state.session_id = new_sid
                    if "sid" in st.query_params:
                        del st.query_params["sid"]
                    st.session_state.current_chat = []
                    st.session_state.vector_store = None
                    st.session_state.site_metadata = None
                    st.session_state.selected_chat_index = None
                    st.session_state.main_quick_url_input = ""
                    st.rerun()

            st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'/>", unsafe_allow_html=True)

            # --- Session History & Filter ---
            db_sessions = session_repository.list_sessions(
                email=user_email,
                guest_client_id=client_id,
                limit=40,
            )

            st.markdown(
                f"""
                <div class="sidebar-section-header">
                    <span>Chat History ({len(db_sessions)})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            session_filter = st.text_input(
                "Filter sessions",
                placeholder="Search history...",
                label_visibility="collapsed",
                key="sidebar_session_filter_input",
            )

            filtered_sessions = db_sessions
            if session_filter and session_filter.strip():
                kw = session_filter.strip().lower()
                filtered_sessions = [
                    s for s in db_sessions
                    if kw in (s.get("title") or "").lower() or kw in (s.get("url") or "").lower()
                ]

            if not filtered_sessions:
                st.caption("No conversations found.")
            else:
                for s in filtered_sessions:
                    s_id = s["session_id"]
                    raw_title = (s.get("title") or s.get("url") or "Untitled Chat").strip()
                    display_title = raw_title[:26] + "..." if len(raw_title) > 26 else raw_title
                    active = (st.session_state.get("session_id") == s_id)
                    btn_type = "primary" if active else "secondary"

                    col_sess, col_del = st.columns([5.2, 1.2], gap="small", vertical_alignment="center")
                    with col_sess:
                        if st.button(display_title, key=f"dbsess_btn_{s_id}", use_container_width=True, type=btn_type):
                            full_s = session_repository.get_session(s_id)
                            if full_s:
                                st.session_state.session_id = s_id
                                st.query_params["sid"] = s_id
                                sess_url = full_s.get("url")
                                messages = full_s.get("messages", [])
                                if sess_url:
                                    url_hash = compute_url_hash(sess_url)
                                    cached_vs = vector_store_cache.get(url_hash)
                                    meta = vector_store_cache.get_metadata(url_hash)
                                    if not meta:
                                        cache_entry = url_cache_repository.get_by_hash(url_hash)
                                        if cache_entry:
                                            meta = cache_entry.get("metadata") or {
                                                "url": sess_url,
                                                "title": cache_entry.get("title") or full_s.get("title", "Active Document"),
                                                "word_count": cache_entry.get("word_count", 0),
                                                "paywall_bypassed": cache_entry.get("paywall_bypassed", False),
                                                "vector_session_id": cache_entry.get("vector_session_id"),
                                                "session_id": s_id,
                                            }
                                        else:
                                            meta = {
                                                "url": sess_url,
                                                "title": full_s.get("title", "Active Document"),
                                                "word_count": 0,
                                                "paywall_bypassed": False,
                                                "session_id": s_id,
                                            }
                                    if not cached_vs:
                                        vec_sid = meta.get("vector_session_id") or s_id
                                        cached_vs = rag_service.get_session_index(vec_sid, metadata=meta)
                                    st.session_state.vector_store = cached_vs
                                    st.session_state.site_metadata = meta
                                else:
                                    st.session_state.vector_store = None
                                    st.session_state.site_metadata = None
                                st.session_state.current_chat = messages
                                st.rerun()

                    with col_del:
                        if st.button("🗑️", key=f"dbsess_del_{s_id}", help="Delete this session"):
                            try:
                                session_repository.delete_session(s_id)
                                if st.session_state.get("session_id") == s_id:
                                    new_sid = str(uuid.uuid4())
                                    st.session_state.session_id = new_sid
                                    if "sid" in st.query_params:
                                        del st.query_params["sid"]
                                    st.session_state.current_chat = []
                                    st.session_state.vector_store = None
                                    st.session_state.site_metadata = None
                                st.rerun()
                            except Exception as del_err:
                                st.error(f"Delete failed: {del_err}")

            # --- Compact Quota Footer ---
            st.markdown("<div style='flex-grow: 1; min-height: 16px;'></div>", unsafe_allow_html=True)
            quota_status = {}
            try:
                quota_status = user_service.get_quota_status(email=user_email, client_id=client_id)
            except Exception:
                pass

            remaining = quota_status.get("requests_remaining", 50) if quota_status else 50
            daily_limit = quota_status.get("daily_limit", 50) if quota_status else 50
            bar_color = "#34d399" if remaining > 10 else ("#facc15" if remaining > 3 else "#ef4444")

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 4px; border-top: 1px solid rgba(255,255,255,0.06);">
                    <span style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">
                        Queries
                    </span>
                    <span style="font-size: 11px; font-weight: 700; color: {bar_color}; font-family: monospace;">
                        {remaining} / {daily_limit}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.sidebar.error(f"Sidebar error: {e}")
