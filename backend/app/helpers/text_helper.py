import re

from backend.app.core.logging import logger


def count_words(text: str) -> int:
    """Accurately calculates total words in text."""
    try:
        if not text:
            return 0
        words = re.findall(r"\b\w+\b", text)
        return len(words)
    except Exception as e:
        logger.error(f"Error counting words: {e}", exc_info=True)
        return 0


def calculate_reading_time(word_count: int, wpm: int = 200) -> int:
    """Estimates reading time in minutes based on words-per-minute."""
    try:
        if not word_count or word_count <= 0:
            return 0
        wpm_rate = wpm if wpm > 0 else 200
        return max(1, round(word_count / wpm_rate))
    except Exception as e:
        logger.error(f"Error calculating reading time: {e}", exc_info=True)
        return 0


def truncate_text(text: str, max_length: int = 200, ellipsis: str = "...") -> str:
    """Truncates text safely at word boundary with an ellipsis."""
    try:
        if not text or len(text) <= max_length:
            return text or ""
        truncated = text[:max_length].rsplit(" ", 1)[0]
        return truncated + ellipsis
    except Exception as e:
        logger.error(f"Error truncating text: {e}", exc_info=True)
        return text[:max_length] if text else ""


def clean_whitespace(text: str) -> str:
    """Normalizes multiple consecutive whitespace characters and blank lines."""
    try:
        if not text:
            return ""
        # Replace multiple spaces with a single space
        cleaned = re.sub(r"[ \t]+", " ", text)
        # Replace more than 2 consecutive newlines with 2 newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    except Exception as e:
        logger.error(f"Error cleaning whitespace: {e}", exc_info=True)
        return text.strip() if text else ""
