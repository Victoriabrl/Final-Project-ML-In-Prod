"""Configuration read from environment variables (12-factor III). No secret has a default."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # None = let MLflow read MLFLOW_TRACKING_URI itself (credentials are read by MLflow
    # from MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD, never stored here).
    mlflow_tracking_uri: str | None
    model_name: str
    model_alias: str
    require_approved: bool
    retry_initial_delay: float
    retry_max_delay: float
    # Baked into the image at build time (Dockerfile ARG GIT_SHA); None outside an image.
    git_sha: str | None = None
    # Live drift: window of recent predictions, recomputed every N predictions.
    drift_window_size: int = 500
    drift_recompute_every: int = 100


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    """Read the environment at call time (not import time) so tests can override it."""
    return Settings(
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or None,
        model_name=os.environ.get("MODEL_NAME", "sms-spam"),
        model_alias=os.environ.get("MODEL_ALIAS", "champion"),
        require_approved=_bool(os.environ.get("REQUIRE_APPROVED", "true")),
        retry_initial_delay=float(os.environ.get("MODEL_LOAD_RETRY_INITIAL_DELAY", "2")),
        retry_max_delay=float(os.environ.get("MODEL_LOAD_RETRY_MAX_DELAY", "60")),
        git_sha=os.environ.get("GIT_SHA") or None,
        drift_window_size=int(os.environ.get("DRIFT_WINDOW_SIZE", "500")),
        drift_recompute_every=int(os.environ.get("DRIFT_RECOMPUTE_EVERY", "100")),
    )
