import logging
import sys
import warnings
from pathlib import Path

# Suppress google_genai SDK automatic function calling warning
warnings.filterwarnings("ignore", message=".*Automatic Function Calling.*")
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.app.api.agent_routes import router as agent_router
from backend.app.api.chat_routes import router as chat_router
from backend.app.api.frontend_routes import FRONTEND_DIR, router as frontend_router
from backend.app.api.user_routes import router as user_router
from backend.app.cache.vector_cache import vector_store_cache
from backend.app.core.config import settings
from backend.app.core.errors import (
    WebChatException,
    generic_exception_handler,
    webchat_exception_handler,
)
from backend.app.core.logging import logger
from backend.app.core.rate_limiter import RateLimitMiddleware
from backend.app.core.security import SecurityHeadersMiddleware
from backend.app.dtos.common_dto import HealthResponseDto
from backend.app.helpers.service_launcher import launch_streamlit
from backend.app.services.memory_service import memory_service
from backend.config.database import init_db

# Add project root to sys.path for backend.* imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Environment and directory paths
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent

_backend_env = _current_dir / ".env"
_root_env = _root_dir / ".env"
if _backend_env.exists():
    load_dotenv(_backend_env, override=False)
if _root_env.exists():
    load_dotenv(_root_env, override=False)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management: startup diagnostics, database init & clean shutdown."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Backend host: {settings.BACKEND_HOST}:{settings.BACKEND_PORT}")
    logger.info(f"Gemini API: {'Configured' if gemini_key else 'Missing'}")
    logger.info(f"Groq API: {'Configured' if groq_key else 'Missing'}")
    logger.info(f"Fallback Chain Models: {len(settings.DEFAULT_MODELS)}")
    logger.info("=" * 60)

    # Initialize PostgreSQL database tables
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)

    # Build Memori Labs database storage and data structures on PostgreSQL
    try:
        memory_service.build_storage()
    except Exception as e:
        logger.warning(f"Memori Labs storage build warning during startup: {e}")

    # Verify mandatory Qdrant Cloud connection
    try:
        from backend.app.services.qdrant_service import qdrant_service
        _ = qdrant_service.client
        logger.info(f"Qdrant Vector DB: Connected (Mandatory - {settings.QDRANT_URL[:40]}...)")
    except Exception as qdrant_err:
        logger.error(f"Qdrant connection verification failed: {qdrant_err}")
        raise RuntimeError(f"Mandatory Qdrant connection failed: {qdrant_err}") from qdrant_err

    # Pre-warm FlashRank neural reranker model to eliminate query latency
    try:
        from backend.app.services.rerank_service import rerank_service
        _ = rerank_service.ranker
        logger.info("FlashRank Reranker: Pre-warmed & cached in memory.")
    except Exception as rerank_err:
        logger.warning(f"FlashRank reranker pre-warm warning: {rerank_err}")

    # Suppress SDK warning logs
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}...")
    vector_store_cache.clear()
    logger.info("Clean shutdown complete.")


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade AI Web Chat API with paywall bypass, multi-provider LLM failover, and PostgreSQL persistence.",
    lifespan=lifespan,
)

# Custom Security Headers Middleware
# Custom Security & Rate Limiting Middlewares
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handlers
app.add_exception_handler(WebChatException, webchat_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# System Health Check Endpoint
@app.get(f"{settings.API_V1_STR}/health", response_model=HealthResponseDto, tags=["System"])
@app.get("/health", response_model=HealthResponseDto, tags=["System"])
def health_check():
    """Returns system diagnostic status, database readiness, and provider configuration."""
    try:
        return HealthResponseDto(
            status="healthy",
            version=settings.VERSION,
            gemini_configured=bool(os.getenv("GEMINI_API_KEY")),
            groq_configured=bool(os.getenv("GROQ_API_KEY")),
            active_models_count=len(settings.DEFAULTMODELS if hasattr(settings, "DEFAULTMODELS") else settings.DEFAULT_MODELS),
        )
    except Exception as e:
        logger.error(f"Error in health_check endpoint: {e}", exc_info=True)
        return HealthResponseDto(
            status="unhealthy",
            version=settings.VERSION,
            gemini_configured=False,
            groq_configured=False,
            active_models_count=0,
        )


# Mount Domain Routers FIRST (Agent, Chat, Users & Frontend)
app.include_router(agent_router, prefix=settings.API_V1_STR)
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(user_router, prefix=settings.API_V1_STR)
app.include_router(frontend_router)

# Mount Frontend Static Assets AFTER Routers
if FRONTEND_DIR.exists():
    dist_dir = FRONTEND_DIR / "dist"
    if dist_dir.exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")
        app.mount("/static", StaticFiles(directory=str(dist_dir), html=True), name="static")
    else:
        # Fallback to serving the legacy frontend if dist is missing
        css_dir = FRONTEND_DIR / "css"
        js_dir = FRONTEND_DIR / "js"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="frontend_css")
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="frontend_js")



def main():
    """Parses CLI arguments and launches FastAPI backend service or Streamlit dashboard."""
    try:
        parser = argparse.ArgumentParser(
            description="WebChat AI Backend & Service Launcher",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            "service",
            nargs="?",
            default="backend",
            choices=["backend", "api", "streamlit"],
            help=(
                "Service to launch:\n"
                "  backend   : Launch FastAPI REST/SSE backend server (default)\n"
                "  api       : Alias for backend\n"
                "  streamlit : Launch modern Streamlit web interface\n"
            ),
        )
        parser.add_argument("--host", default=None, help=f"Host address for backend (default: {settings.BACKEND_HOST})")
        parser.add_argument("--port", type=int, default=None, help="Port for service")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload for uvicorn")

        args, unknown = parser.parse_known_args()

        if args.service in ["backend", "api"]:
            host = args.host or settings.BACKEND_HOST
            port = args.port or settings.BACKEND_PORT
            logger.info(f"Starting FastAPI backend on http://{host}:{port}...")
            uvicorn.run(
                "backend.main:app",
                host=host,
                port=port,
                reload=args.reload,
            )
        elif args.service == "streamlit":
            launch_streamlit(port=args.port)
    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
