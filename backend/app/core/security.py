import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response



class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds essential production security headers to HTTP responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def sanitize_input(text: str) -> str:
    """Sanitizes user input string to remove harmful null bytes and control characters."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters (except newline, carriage return, tab)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return sanitized.strip()
