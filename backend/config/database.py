import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.logging import logger
from backend.app.models.base import Base
import backend.app.models.models  # noqa: F401

# Load environment variables reliably from backend/.env or root/.env
_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent
_root_dir = _backend_dir.parent

for candidate_env in [_backend_dir / ".env", _root_dir / ".env", Path.cwd() / ".env"]:
    if candidate_env.exists():
        load_dotenv(candidate_env, override=False)

# Quota & Rate Limit Settings
GUEST_MAX_REQUESTS = 5
GUEST_MAX_REQUESTS = 50
USER_DAILY_REQUEST_LIMIT = 50

# Re-export Base for convenience
Base = Base

# Global Engine and Sessionmaker placeholders
engine = None
SessionLocal = None


def build_database_url() -> str:
    """Constructs PostgreSQL database connection string from environment variables."""
    try:
        # Direct DATABASE_URL
        direct_url = os.getenv("DATABASE_URL")
        if direct_url:
            return direct_url.strip()

        # Individual DB credentials (e.g., Aiven PostgreSQL)
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        ssl_mode = os.getenv("SSL_MODE", "require")

        if db_host and db_name:
            auth = f"{db_user}:{db_password}@" if db_user else ""
            port = f":{db_port}" if db_port else ""
            ssl = f"?sslmode={ssl_mode}" if ssl_mode else ""
            return f"postgresql://{auth}{db_host}{port}/{db_name}{ssl}"

        return ""
    except Exception as e:
        logger.error(f"Error building database URL: {e}", exc_info=True)
        return ""


DATABASE_URL = build_database_url()


def get_engine():
    """Initializes and returns SQLAlchemy PostgreSQL database engine."""
    global engine, SessionLocal
    try:
        if engine is not None:
            return engine

        db_url = build_database_url()
        if not db_url:
            raise ValueError(
                "PostgreSQL database is not configured. Please set DB_USER, DB_PASSWORD, DB_HOST, and DB_NAME in backend/.env"
            )

        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL database successfully.")

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine
    except Exception as e:
        logger.error(f"PostgreSQL database connection error: {e}", exc_info=True)
        raise


def init_db():
    """Initializes database tables on application startup."""
    """Initializes database tables and runs schema migrations on application startup."""
    global engine, SessionLocal
    try:
        active_engine = get_engine()
        Base.metadata.create_all(bind=active_engine)
        logger.info("Database tables initialized successfully on PostgreSQL.")

        # Run safe non-destructive schema migrations
        with active_engine.connect() as conn:
            try:
                from sqlalchemy import text
                conn.execute(text("ALTER TABLE webchat_guest_usage ADD COLUMN IF NOT EXISTS ip_address VARCHAR(64);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_webchat_guest_usage_ip_address ON webchat_guest_usage(ip_address);"))
                conn.commit()
            except Exception as mig_err:
                logger.warning(f"Database schema migration warning: {mig_err}")

        logger.info("Database tables and schema initialized successfully on PostgreSQL.")
    except Exception as e:
        logger.error(f"Database table initialization failed on PostgreSQL: {e}", exc_info=True)
        raise


@contextmanager
def get_db_session():
    """Context manager for thread-safe database transactions."""
    if SessionLocal is None:
        get_engine()
    session: Session = SessionLocal()  # type: ignore
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency yielding database session."""
    if SessionLocal is None:
        get_engine()
    session: Session = SessionLocal()  # type: ignore
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
