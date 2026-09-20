import hashlib
import hmac
import os
import re
from typing import List, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from backend.app.core.config import settings
from backend.app.core.logging import logger


def validate_url(url: str) -> Tuple[bool, str]:
    """Validates target URL format, length, and HTTP/HTTPS protocol."""
    try:
        if not url or not isinstance(url, str):
            return False, "Please provide a valid URL."

        clean_url = url.strip()
        if len(clean_url) > 2048:
            return False, "URL length exceeds maximum limit of 2048 characters."

        parsed = urlparse(clean_url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format. Please include http:// or https://"

        if parsed.scheme.lower() not in ["http", "https"]:
            return False, "URL protocol must be HTTP or HTTPS."

        return True, ""
    except Exception as e:
        logger.error(f"Error validating URL '{url}': {e}", exc_info=True)
        return False, f"Invalid URL format: {e}"


# Tracking query parameters to drop for deterministic canonicalization
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "source",
}


def normalize_url(url: str) -> str:
    """Normalizes URL by lowercasing domain, removing tracking params, and stripping trailing slashes."""
    try:
        if not url:
            return ""
        clean = url.strip()
        if not clean.lower().startswith("http://") and not clean.lower().startswith("https://"):
            clean = f"https://{clean}"

        parsed = urlparse(clean)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Path normalization: strip trailing slash unless root path
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        if not path:
            path = "/"

        # Query param normalization: remove tracking params and sort remaining
        query_params = []
        if parsed.query:
            for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                if k.lower() not in TRACKING_PARAMS:
                    query_params.append((k, v))
            query_params.sort(key=lambda x: x[0])
        new_query = urlencode(query_params) if query_params else ""

        canonical = urlunparse((scheme, netloc, path, parsed.params, new_query, ""))
        return canonical
    except Exception as e:
        logger.error(f"Error normalizing URL '{url}': {e}", exc_info=True)
        return url.strip() if url else ""


def compute_url_hash(
    url_or_urls: Union[str, List[str]],
    secret: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Computes a deterministic, cryptographically keyed HMAC hash from URLs using environment settings."""
    try:
        if isinstance(url_or_urls, str):
            raw_list = [u.strip() for u in re.split(r"[\n,]+", url_or_urls) if u.strip()]
        else:
            raw_list = list(url_or_urls)

        normalized_set = {normalize_url(u) for u in raw_list if u.strip()}
        canonical_str = "||".join(sorted(normalized_set))

        # Read secret key purely from environment variables or settings
        secret_key = secret or os.getenv("URL_HASH_SECRET") or settings.URL_HASH_SECRET

        if not secret_key:
            raise ValueError(
                "URL_HASH_SECRET must be set in your .env file. "
                "No hardcoded secret keys are permitted in code."
            )

        # Read hashing algorithm purely from environment variables or settings
        algo_name = algorithm or os.getenv("URL_HASH_ALGORITHM") or settings.URL_HASH_ALGORITHM

        if not algo_name:
            raise ValueError(
                "URL_HASH_ALGORITHM must be set in your .env file. "
                "No hardcoded hash algorithms are permitted in code."
            )

        return hmac.new(
            secret_key.encode("utf-8"),
            canonical_str.encode("utf-8"),
            digestmod=algo_name.strip().lower(),
        ).hexdigest()
    except Exception as e:
        logger.error(f"Error computing URL hash: {e}", exc_info=True)
        raise


def extract_domain(url: str) -> str:
    """Extracts base domain from a target URL."""
    try:
        if not url:
            return ""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception as e:
        logger.error(f"Error extracting domain from '{url}': {e}", exc_info=True)
        return ""
