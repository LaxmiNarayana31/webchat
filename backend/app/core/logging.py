import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Optional

# Ensure standard output uses UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import warnings

# Suppress noisy google_genai automatic function calling (AFC) recommendation warnings
warnings.filterwarnings("ignore", message=".*Automatic Function Calling.*")
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Determine project root and default logs directory
_current_file = Path(__file__).resolve()
_backend_dir = _current_file.parent.parent.parent
_project_root = _backend_dir.parent
DEFAULT_LOGS_DIR = _project_root / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOGS_DIR / "webchat.log"


def setup_logging(
    log_level: Optional[str] = None,
    log_file_path: Optional[str] = None,
) -> logging.Logger:
    """Configures structured logging for the application.

    - Our code-level logs ('webchat'): Routine INFO/DEBUG logs are stored in the
      log file (logs/webchat.log) to keep the terminal clean, while ERRORs always
      print to terminal stderr.
    - Package-level logs & errors: Third-party packages (Memori, FAISS, DB drivers,
      etc.) are NOT suppressed. Package errors and outputs remain visible.
    """
    try:
        level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)

        log_format = "%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        # Resolve log file path and ensure directory exists
        file_path_str = log_file_path or os.getenv("LOG_FILE", str(DEFAULT_LOG_FILE))
        log_path = Path(file_path_str)
        if not log_path.is_absolute():
            log_path = _project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating File Handler: records all application and package events (10MB max, 5 backups)
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Console Handler for package-level errors and warnings (packages are NOT suppressed)
        package_console_handler = logging.StreamHandler(sys.stderr)
        package_console_handler.setLevel(logging.WARNING)
        package_console_handler.setFormatter(formatter)

        # Configure root logger: records to file, alerts on package errors/warnings
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(package_console_handler)

        # Configure our code-level application logger ('webchat'):
        # 1. Writes all INFO/DEBUG/WARNING/ERROR to logs/webchat.log
        # 2. Only prints ERRORs to terminal stderr (preventing routine startup clutter)
        # 3. Does not duplicate or leak INFO clutter to terminal
        app_logger = logging.getLogger("webchat")
        app_logger.setLevel(level)
        app_logger.handlers.clear()
        app_logger.addHandler(file_handler)

        app_console_handler = logging.StreamHandler(sys.stderr)
        app_console_handler.setLevel(logging.ERROR)
        app_console_handler.setFormatter(formatter)
        app_logger.addHandler(app_console_handler)

        app_logger.propagate = False

        return app_logger
    except Exception:
        return logging.getLogger("webchat")


logger = setup_logging()
