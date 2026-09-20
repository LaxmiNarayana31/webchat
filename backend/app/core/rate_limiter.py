from collections import defaultdict
from dataclasses import dataclass, field
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.core.logging import logger


def get_client_ip(request: Request) -> str:
    """Safely extracts the real client IP address from proxy headers or socket connection.
    
    Handles X-Forwarded-For (extracts leftmost client IP), X-Real-IP, and direct socket host.
    """
    try:
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            ips = [ip.strip() for ip in x_forwarded_for.split(",") if ip.strip()]
            if ips:
                return ips[0]

        x_real_ip = request.headers.get("x-real-ip")
        if x_real_ip and x_real_ip.strip():
            return x_real_ip.strip()

        if request.client and request.client.host:
            return request.client.host.strip()

        return "127.0.0.1"
    except Exception as e:
        logger.warning(f"Error extracting client IP: {e}")
        return "127.0.0.1"


# -----------------------------------------------------------------------------
# LLM Provider Model Rate Limiter (Token Bucket & Sliding Tracker)
# -----------------------------------------------------------------------------
class TokenBucket:
    """Thread-safe Token Bucket rate limiter for LLM providers."""

    def __init__(self, capacity: int, refill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = refill_rate_per_sec
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> bool:
        """Attempts to acquire tokens from the bucket, returning True if successful."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_remaining(self) -> float:
        """Calculates remaining token capacity after simulated refill."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(self.capacity, self.tokens + elapsed * self.refill_rate)


class SlidingWindowTracker:
    """Tracks request timestamps within a sliding time window for LLM telemetry."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.timestamps = []
        self._lock = threading.Lock()

    def record_request(self) -> int:
        """Records timestamp of a new request and returns total within the window."""
        with self._lock:
            now = time.time()
            self._purge(now)
            self.timestamps.append(now)
            return len(self.timestamps)

    def current_count(self) -> int:
        """Returns number of requests recorded in the current active window."""
        with self._lock:
            self._purge(time.time())
            return len(self.timestamps)

    def _purge(self, now: float):
        """Removes timestamps that fall outside the sliding window."""
        cutoff = now - self.window_seconds
        self.timestamps = [ts for ts in self.timestamps if ts > cutoff]


class ModelRateLimiter:
    """Coordinates rate limiting, backoff, and cooldowns across model providers."""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._trackers: Dict[str, SlidingWindowTracker] = {}
        self._cooldowns: Dict[str, float] = {}
        self._failures: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _get_key(self, provider: str, model: str) -> str:
        return f"{provider.lower()}:{model.lower()}"

    def _ensure_limiter(self, provider: str, model: str, rpm_limit: int):
        key = self._get_key(provider, model)
        with self._lock:
            if key not in self._buckets:
                refill_rate = max(0.1, rpm_limit / 60.0)
                self._buckets[key] = TokenBucket(capacity=rpm_limit, refill_rate_per_sec=refill_rate)
                self._trackers[key] = SlidingWindowTracker(window_seconds=60.0)
                self._failures[key] = 0

    def can_proceed(self, provider: str, model: str, rpm_limit: int = 60) -> Tuple[bool, str]:
        key = self._get_key(provider, model)
        self._ensure_limiter(provider, model, rpm_limit)

        now = time.time()
        with self._lock:
            cooldown_expiry = self._cooldowns.get(key, 0)
            if now < cooldown_expiry:
                remaining_cooldown = int(cooldown_expiry - now)
                return False, f"Model in cooldown for {remaining_cooldown}s"

        bucket = self._buckets[key]
        if not bucket.acquire(1.0):
            return False, "Rate limit reached (token bucket empty)"

        self._trackers[key].record_request()
        return True, "OK"

    def record_success(self, provider: str, model: str):
        key = self._get_key(provider, model)
        with self._lock:
            self._failures[key] = 0

    def record_rate_limit_error(self, provider: str, model: str, retry_after: Optional[int] = None):
        key = self._get_key(provider, model)
        with self._lock:
            failures = self._failures.get(key, 0) + 1
            self._failures[key] = failures

            if retry_after and retry_after > 0:
                cooldown_sec = float(retry_after)
            else:
                base = min(60.0, (2.0 ** failures))
                jitter = random.uniform(0.5, 2.0)
                cooldown_sec = base + jitter

            self._cooldowns[key] = time.time() + cooldown_sec
            logger.warning(
                f"Rate limit triggered for {key}. Consecutive failures: {failures}. Cooldown: {cooldown_sec:.1f}s"
            )

    def calculate_retry_delay(self, attempt: int, base_delay: float = 1.0, max_delay: float = 10.0) -> float:
        temp = min(max_delay, base_delay * (2 ** attempt))
        return random.uniform(0.5, temp)

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        status_data = {}
        with self._lock:
            for key, tracker in self._trackers.items():
                bucket = self._buckets.get(key)
                cooldown_expiry = self._cooldowns.get(key, 0)
                in_cooldown = now < cooldown_expiry
                remaining_cd = max(0.0, cooldown_expiry - now) if in_cooldown else 0.0

                status_data[key] = {
                    "requests_last_minute": tracker.current_count(),
                    "tokens_available": round(bucket.get_remaining(), 1) if bucket else 0,
                    "in_cooldown": in_cooldown,
                    "cooldown_remaining_sec": round(remaining_cd, 1),
                    "consecutive_failures": self._failures.get(key, 0),
                }
        return status_data


global_rate_limiter = ModelRateLimiter()


# -----------------------------------------------------------------------------
# Anti-DDoS Burst & IP Security Rate Limiter
# -----------------------------------------------------------------------------
@dataclass
class SlidingWindowBucket:
    """Tracks timestamps of incoming requests in a sliding window."""
    timestamps: List[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_allowed(self, max_requests: int, window_seconds: float) -> Tuple[bool, int, float]:
        now = time.time()
        window_start = now - window_seconds

        with self.lock:
            self.timestamps = [t for t in self.timestamps if t > window_start]
            if len(self.timestamps) >= max_requests:
                oldest = self.timestamps[0] if self.timestamps else now
                retry_after = max(1.0, (oldest + window_seconds) - now)
                return False, len(self.timestamps), retry_after

            self.timestamps.append(now)
            return True, len(self.timestamps), 0.0


class SecurityRateLimiter:
    """Enterprise-grade anti-DDoS, burst protection, and endpoint rate limiter."""

    def __init__(self):
        self._burst_buckets: Dict[str, SlidingWindowBucket] = defaultdict(SlidingWindowBucket)
        self._heavy_buckets: Dict[str, SlidingWindowBucket] = defaultdict(SlidingWindowBucket)
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = time.time()

    def _maybe_cleanup(self):
        now = time.time()
        if now - self._last_cleanup < 300:
            return

        with self._cleanup_lock:
            if now - self._last_cleanup < 300:
                return
            self._last_cleanup = now
            threshold = now - 3600

            for store in [self._burst_buckets, self._heavy_buckets]:
                stale_keys = [
                    k for k, b in store.items()
                    if not b.timestamps or b.timestamps[-1] < threshold
                ]
                for k in stale_keys:
                    store.pop(k, None)

    def check_burst(self, ip: str, max_requests: int = 10, window_seconds: float = 5.0) -> Tuple[bool, float]:
        self._maybe_cleanup()
        bucket = self._burst_buckets[ip]
        allowed, count, retry_after = bucket.is_allowed(max_requests=max_requests, window_seconds=window_seconds)
        return allowed, retry_after

    def check_heavy_endpoint(self, ip: str, max_requests: int = 8, window_seconds: float = 60.0) -> Tuple[bool, float]:
        self._maybe_cleanup()
        bucket = self._heavy_buckets[ip]
        allowed, count, retry_after = bucket.is_allowed(max_requests=max_requests, window_seconds=window_seconds)
        return allowed, retry_after


security_rate_limiter = SecurityRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware that enforces anti-DDoS burst protection across all API routes."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Whitelist static assets, health check, docs, and favicon
        if (
            path.startswith("/assets")
            or path.startswith("/css")
            or path.startswith("/js")
            or path.startswith("/static")
            or path in ["/health", "/api/health", "/docs", "/openapi.json", "/favicon.ico"]
            or request.method == "OPTIONS"
        ):
            return await call_next(request)

        client_ip = get_client_ip(request)

        # 1. Anti-DDoS Burst Protection (Drop rapid floods immediately)
        burst_allowed, retry_after = security_rate_limiter.check_burst(
            ip=client_ip,
            max_requests=12,
            window_seconds=5.0,
        )

        if not burst_allowed:
            logger.warning(f"Anti-DDoS Shield: Dropped burst flood from IP '{client_ip}' on '{path}' (Retry after: {retry_after:.1f}s)")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": {
                        "message": "Too many requests. Anti-DDoS rate limit exceeded.",
                        "retry_after_seconds": round(retry_after, 1),
                        "type": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        # 2. Heavy Endpoint Scrape/Crawl Protection
        if path in ["/api/scrape", "/api/crawl", "/scrape", "/crawl"]:
            heavy_allowed, heavy_retry = security_rate_limiter.check_heavy_endpoint(
                ip=client_ip,
                max_requests=8,
                window_seconds=60.0,
            )
            if not heavy_allowed:
                logger.warning(f"Anti-DDoS Shield: Heavy endpoint limit reached for IP '{client_ip}' on '{path}'")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": {
                            "message": "Scrape/Crawl rate limit exceeded. Please wait before indexing new websites.",
                            "retry_after_seconds": round(heavy_retry, 1),
                            "type": "endpoint_rate_limit_exceeded",
                        }
                    },
                    headers={"Retry-After": str(int(heavy_retry) + 1)},
                )

        return await call_next(request)
