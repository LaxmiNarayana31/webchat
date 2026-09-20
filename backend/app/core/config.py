import json
from typing import Any, Dict, List

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env"),
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Project Information
    PROJECT_NAME: str = "WebChat AI"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/webchat.log", alias="LOG_FILE")

    # Server Settings
    BACKEND_HOST: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    BACKEND_PORT: int = Field(default=8000, alias="BACKEND_PORT")
    ALLOWED_ORIGINS: Any = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "*",
        ],
        alias="ALLOWED_ORIGINS",
    )

    @property
    def cors_origins(self) -> List[str]:
        """Safely parses allowed origins from string, JSON array, or list."""
        v = self.ALLOWED_ORIGINS
        if isinstance(v, str):
            v_clean = v.strip()
            if v_clean.startswith("[") and v_clean.endswith("]"):
                try:
                    return json.loads(v_clean)
                except Exception:
                    pass
            return [origin.strip() for origin in v_clean.split(",") if origin.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return ["*"]

    # Scraper Settings
    USER_AGENT: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        ),
        alias="USER_AGENT",
    )
    SCRAPER_TIMEOUT: int = Field(default=15, alias="SCRAPER_TIMEOUT")
    JINA_READER_ENABLED: bool = Field(default=True, alias="JINA_READER_ENABLED")
    ARCHIVE_FALLBACK_ENABLED: bool = Field(default=True, alias="ARCHIVE_FALLBACK_ENABLED")

    # Vector DB (Qdrant) Settings - Strictly Mandatory
    QDRANT_URL: str = Field(default="", alias="QDRANT_URL")
    QDRANT_API_KEY: str = Field(default="", alias="QDRANT_API_KEY")

    # Upstash Redis & Semantic Cache Settings
    REDIS_URL: str = Field(default="", alias="REDIS_URL")
    UPSTASH_REDIS_REST_URL: str = Field(default="", alias="UPSTASH_REDIS_REST_URL")
    UPSTASH_REDIS_REST_TOKEN: str = Field(default="", alias="UPSTASH_REDIS_REST_TOKEN")
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_TTL: int = 604800  # 7 days in seconds

    # PostgreSQL / DB Settings
    DB_USER: str = Field(default="", alias="DB_USER")
    DB_PASSWORD: str = Field(default="", alias="DB_PASSWORD")
    DB_HOST: str = Field(default="", alias="DB_HOST")
    DB_PORT: int = Field(default=5432, alias="DB_PORT")
    DB_NAME: str = Field(default="defaultdb", alias="DB_NAME")

    # Security & Hashing (strictly configured via environment variables, zero hardcoded values)
    URL_HASH_ALGORITHM: str = Field(
        default="sha256",
        alias="URL_HASH_ALGORITHM",
    )
    URL_HASH_SECRET: str = Field(
        default="",
        alias="URL_HASH_SECRET",
    )

    # Rate Limiting & Resiliency
    GEMINI_RPM_LIMIT: int = Field(default=60, alias="GEMINI_RPM_LIMIT")
    GROQ_RPM_LIMIT: int = Field(default=30, alias="GROQ_RPM_LIMIT")
    RETRY_MAX_ATTEMPTS: int = Field(default=3, alias="RETRY_MAX_ATTEMPTS")
    COOLDOWN_BASE_SECONDS: float = Field(default=15.0, alias="COOLDOWN_BASE_SECONDS")

    # Priority Multi-Model Fallback Chain (Google Flash Models and Groq Models)
    DEFAULT_MODELS: List[Dict[str, str]] = [
        {"provider": "gemini", "model": "gemini-3.6-flash"},
        {"provider": "groq", "model": "openai/gpt-oss-120b"},
        {"provider": "gemini", "model": "gemini-3.5-flash"},
        {"provider": "groq", "model": "groq/compound"},
        {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
        {"provider": "groq", "model": "openai/gpt-oss-20b"},
        {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
        {"provider": "groq", "model": "groq/compound-mini"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    ]

    @model_validator(mode="after")
    def validate_mandatory_services(self) -> "Settings":
        """Enforces that strictly mandatory services like Qdrant have configured credentials."""
        if not self.QDRANT_URL or not str(self.QDRANT_URL).strip():
            raise ValueError(
                "QDRANT_URL is mandatory and cannot be empty. "
                "Please configure QDRANT_URL in backend/.env"
            )
        return self


settings = Settings()