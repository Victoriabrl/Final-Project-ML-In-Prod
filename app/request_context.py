"""Per-request context: X-Request-ID and the ``request_completed`` log line.

The id comes from the caller's ``X-Request-ID`` header (the web page and replay.py send
one) or is generated. It is bound to the structlog context, so every line logged while
serving the request carries it, and it is sent back in the response header.
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from fastapi import FastAPI, Request
from starlette.routing import Match

from app.logging_config import get_logger

REQUEST_ID_HEADER = "X-Request-ID"
# Accept only short, safe ids: anything else (too long, spaces, newlines) is replaced,
# so a client cannot inject content into the logs.
_VALID_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
UNMATCHED_PATH = "<unmatched>"

log = get_logger("app.http")


def request_id_from(request: Request) -> str:
    candidate = request.headers.get(REQUEST_ID_HEADER, "")
    return candidate if _VALID_ID.fullmatch(candidate) else uuid.uuid4().hex


def route_path(request: Request) -> str:
    """The route template (``/predict``), never the raw URL: low cardinality, no query."""
    route = request.scope.get("route")
    if route is not None:
        return route.path
    for candidate in request.app.router.routes:
        if candidate.matches(request.scope)[0] == Match.FULL:
            return candidate.path
    return UNMATCHED_PATH


def install_request_context(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request_id_from(request)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request_completed",
                method=request.method,
                path=route_path(request),
                status=500,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        status = response.status_code
        fields = {
            "method": request.method,
            "path": route_path(request),
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if status >= 500:
            log.error("request_completed", **fields)
        else:
            log.info("request_completed", **fields)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
