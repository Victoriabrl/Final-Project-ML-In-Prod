"""Structured JSON logs on stdout (12-factor XI): one JSON object per line.

- Our events go through structlog (``get_logger``); standard-library loggers (uvicorn,
  MLflow, our older ``logging`` calls) are rendered by the same processors, so every
  line the service prints is JSON.
- ``merge_contextvars`` adds the fields bound for the current request (``request_id``),
  so all the lines of one request can be found with a single id.
- Level from ``LOG_LEVEL`` (default INFO). Never log the SMS text or a credential.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, TextIO

import structlog

# uvicorn and MLflow install their own handlers (plain text, propagate=False):
# route them to ours instead, so their lines are JSON too.
_FOREIGN_LOGGERS = ("uvicorn", "uvicorn.error", "mlflow")
# Replaced by our request_completed line (which has the request_id): always silenced.
_SILENCED_LOGGERS = ("uvicorn.access",)

# Safety net: even if a library puts a secret in an error message, it is masked.
SECRET_ENV_VARS = (
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_SAS_TOKEN",
    "RENDER_DEPLOY_HOOK_URL",
    "GF_SMTP_PASSWORD",
    "SMTP_PASSWORD",
)
_URL_CREDENTIALS = re.compile(r"://[^/@\s:]+:[^/@\s]+@")
MASK = "***"


def _redact(value: str, secrets: list[str]) -> str:
    for secret in secrets:
        value = value.replace(secret, MASK)
    return _URL_CREDENTIALS.sub(f"://{MASK}@", value)


def redact_secrets(_logger: Any, _method: str, event_dict: dict) -> dict:
    secrets = [v for name in SECRET_ENV_VARS if len(v := os.environ.get(name, "")) >= 4]
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = _redact(value, secrets)
    return event_dict


def ensure_request_id(_logger: Any, _method: str, event_dict: dict) -> dict:
    """Every line has a request_id key: null for lines outside a request (startup...)."""
    event_dict.setdefault("request_id", None)
    return event_dict


def _shared_processors() -> list:
    return [
        structlog.contextvars.merge_contextvars,
        ensure_request_id,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]


def configure_logging(level: str | None = None, stream: TextIO | None = None) -> None:
    """Configure structlog + the root logger. Safe to call again (tests reconfigure)."""
    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    numeric = logging.getLevelName(level_name)
    if not isinstance(numeric, int):
        numeric, level_name = logging.INFO, "INFO"

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_shared_processors(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(numeric)
    for name in _FOREIGN_LOGGERS:
        foreign = logging.getLogger(name)
        foreign.handlers.clear()
        foreign.propagate = True
    for name in _SILENCED_LOGGERS:
        silenced = logging.getLogger(name)
        silenced.handlers.clear()
        silenced.propagate = False
    # MLflow: no handler of its own when imported later (it is imported lazily), and
    # no tqdm progress bars on stderr while downloading the model (not JSON).
    os.environ.setdefault("MLFLOW_CONFIGURE_LOGGING", "false")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


def get_logger(name: str = "app") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
