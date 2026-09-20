import os
from pathlib import Path
import sys
import uuid

# Patch Windows asyncio ProactorEventLoop WinError 10054 (ConnectionResetError)
if sys.platform.startswith("win"):
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                # On Windows, shutdown() on an already closed remote socket raises WinError 10054.
                # Complete the necessary socket close and detach cleanup safely.
                try:
                    if hasattr(self, "_sock") and self._sock is not None:
                        self._sock.close()
                        self._sock = None
                    server = getattr(self, "_server", None)
                    if server is not None:
                        server._detach(self)
                        self._server = None
                    self._called_connection_lost = True
                except Exception:
                    pass

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except Exception:
        pass

# Ensure root workspace directory is in Python module search path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
import streamlit as st

# Load local environment variables (for local runs)
backend_env = root_dir / "backend" / ".env"
if backend_env.exists():
    load_dotenv(backend_env, override=False)
load_dotenv(root_dir / ".env", override=False)

# Sync Streamlit Cloud secrets to os.environ (for Streamlit Cloud deployment)
try:
    for sec_key in st.secrets:
        sec_val = st.secrets[sec_key]
        if isinstance(sec_val, (str, int, float, bool)) and sec_key not in os.environ:
            os.environ[sec_key] = str(sec_val)
except Exception:
    pass

from backend.app.cache.vector_cache import vector_store_cache
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.repositories.session_repository import session_repository
from backend.app.repositories.url_cache_repository import url_cache_repository
from backend.app.services.rag_service import rag_service
from backend.config.database import init_db
from streamlit_app.components.chat import render_chat
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.styles import CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="WebChat AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session():
    """Initializes Streamlit session state and restores client/session persistence across browser refreshes."""
    try:
        # 1. Restore or create persistent client_id via st.query_params (mirroring localStorage in React)
        qp_cid = st.query_params.get("cid")
        if qp_cid and str(qp_cid).strip():
            client_id = str(qp_cid).strip()
        elif "client_id" in st.session_state and st.session_state.client_id:
            client_id = st.session_state.client_id
            st.query_params["cid"] = client_id
        else:
            client_id = f"streamlit_{uuid.uuid4().hex[:12]}"
            st.query_params["cid"] = client_id

        st.session_state.client_id = client_id

        if "user_email" not in st.session_state:
            st.session_state.user_email = ""
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = []
        if "selected_chat_index" not in st.session_state:
            st.session_state.selected_chat_index = None
        if "selected_model_override" not in st.session_state:
            st.session_state.selected_model_override = None

        # 2. Check if we need to restore an active session on page refresh
        qp_sid = st.query_params.get("sid")
        target_sid = str(qp_sid).strip() if (qp_sid and str(qp_sid).strip()) else None

        # If no target_sid in query params, find the most recent session for this client
        if not target_sid:
            try:
                recent_list = session_repository.list_sessions(guest_client_id=client_id, limit=1)
                if recent_list and len(recent_list) > 0:
                    target_sid = recent_list[0].get("session_id")
                    if target_sid:
                        st.query_params["sid"] = target_sid
            except Exception:
                pass

        # Re-hydrate the target session if session_state is fresh/empty
        if target_sid:
            st.session_state.session_id = target_sid
            if "current_chat" not in st.session_state or len(st.session_state.current_chat) == 0:
                try:
                    full_s = session_repository.get_session(target_sid)
                    if full_s:
                        st.session_state.current_chat = full_s.get("messages", [])
                        sess_url = full_s.get("url")
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
                                        "session_id": target_sid,
                                    }
                                else:
                                    meta = {
                                        "url": sess_url,
                                        "title": full_s.get("title", "Active Document"),
                                        "word_count": 0,
                                        "paywall_bypassed": False,
                                        "session_id": target_sid,
                                    }
                            if not cached_vs and meta:
                                vec_sid = meta.get("vector_session_id") or target_sid
                                cached_vs = rag_service.get_session_index(vec_sid, metadata=meta)
                            st.session_state.vector_store = cached_vs
                            st.session_state.site_metadata = meta
                except Exception as restore_err:
                    print(f"Error restoring session from DB: {restore_err}", file=sys.stderr)
        else:
            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            if "current_chat" not in st.session_state:
                st.session_state.current_chat = []
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = None
            if "site_metadata" not in st.session_state:
                st.session_state.site_metadata = None

        if "current_chat" not in st.session_state:
            st.session_state.current_chat = []
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        if "site_metadata" not in st.session_state:
            st.session_state.site_metadata = None

    except Exception as e:
        print(f"Error initializing session state: {e}", file=sys.stderr)


@st.cache_resource(show_spinner=False)
def ensure_database_initialized():
    """Initializes database tables exactly once across the Streamlit lifecycle."""
    try:
        init_db()
        return True
    except Exception:
        return False


def main():
    """Streamlit Application Entrypoint."""
    try:
        ensure_database_initialized()
        init_session()
        render_sidebar()
        render_chat()
    except Exception as e:
        st.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
# WebChat AI Application Entrypoint

