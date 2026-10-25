"""Process-local HTTP middleware: rate limiting and request correlation ids."""

import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client-IP rate limiter.

    Process-local: each worker enforces its own window, and the client IP
    must be resolved upstream (e.g. X-Forwarded-For at the proxy) when
    deployed behind a load balancer.
    """

    _MAX_TRACKED_CLIENTS = 10_000

    def __init__(
        self, app, max_requests: int, period_seconds: int
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._hits: dict[str, tuple[float, int]] = {}

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self.max_requests <= 0:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start, count = self._hits.get(client, (now, 0))
        if now - window_start >= self.period_seconds:
            window_start, count = now, 0
        self._hits[client] = (window_start, count + 1)

        if len(self._hits) > self._MAX_TRACKED_CLIENTS:
            cutoff = now - self.period_seconds
            self._hits = {
                ip: hit for ip, hit in self._hits.items() if hit[0] > cutoff
            }

        if count + 1 > self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "message": "Rate limit exceeded",
                    "status_code": 429,
                },
            )
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate an X-Request-ID header (or generate one) for tracing."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
