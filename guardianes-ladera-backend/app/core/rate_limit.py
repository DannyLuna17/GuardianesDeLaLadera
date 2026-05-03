from __future__ import annotations

import time
import threading
from collections import defaultdict

from fastapi import Request

from app.core.exceptions import ApiError


class InMemoryRateLimiter:
    """Simple per-IP sliding-window rate limiter.

    Not suitable for multi-process deployments (use Redis-based limiter instead).
    Sufficient for single-process or low-traffic scenarios.
    """

    def __init__(self, max_requests: int = 5, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _cleanup(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._cleanup(key, now)
            if len(self._hits[key]) >= self.max_requests:
                raise ApiError(
                    429,
                    "rate_limited",
                    "Too many requests. Please try again later.",
                )
            self._hits[key].append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


login_limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
