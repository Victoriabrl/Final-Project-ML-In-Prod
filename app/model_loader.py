"""Resolve the MLflow alias to a concrete version, load that version, track readiness."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from app.drift import PROFILE_ARTIFACT
from app.drift_monitor import DriftMonitor
from app.logging_config import get_logger
from app.settings import Settings

log = get_logger("app.model")

SPAM = "spam"
# Internal self-check run before declaring the service ready.
PROBE_TEXTS = ["WINNER! Claim your free prize now, call 09061701461", "See you at lunch tomorrow?"]


class ModelNotReadyError(RuntimeError):
    """Raised when a prediction is requested before an approved model is loaded."""


@dataclass(frozen=True)
class LoadedModel:
    model: object
    model_name: str
    alias: str
    version: str
    run_id: str | None
    data_version: str | None
    loaded_at: str
    # None when the champion's run has no training profile: drift is then not computed.
    drift: DriftMonitor | None = None


def _spam_probabilities(model, texts: list[str]) -> list[float]:
    proba = model.predict_proba(pd.DataFrame({"text": texts}))
    spam_idx = list(model.classes_).index(SPAM)
    return [float(p) for p in proba[:, spam_idx]]


class ModelLoader:
    """Holds the served model. The swap is atomic: readers always see a complete state."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    # --- loading ----------------------------------------------------------------------

    def load(self) -> LoadedModel:
        """Resolve alias -> version, load models:/<name>/<version>, self-check, swap in."""
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        s = self.settings
        if s.mlflow_tracking_uri:
            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        client = MlflowClient()

        mv = client.get_model_version_by_alias(s.model_name, s.model_alias)
        status = mv.tags.get("status")
        if s.require_approved and status != "approved":
            raise RuntimeError(
                f"{s.model_name} v{mv.version} (alias {s.model_alias!r}) has status "
                f"{status!r}, expected 'approved'"
            )

        # Load the concrete version, not the alias: the served version cannot change
        # behind our back if the alias moves while we download.
        model = mlflow.sklearn.load_model(f"models:/{s.model_name}/{mv.version}")
        probs = _spam_probabilities(model, PROBE_TEXTS)
        if not all(0.0 <= p <= 1.0 for p in probs):
            raise RuntimeError(f"self-check failed: probabilities {probs}")

        drift = self._drift_monitor(mv.run_id, str(mv.version))
        loaded = LoadedModel(
            model=model,
            model_name=s.model_name,
            alias=s.model_alias,
            version=str(mv.version),
            run_id=mv.run_id,
            data_version=mv.tags.get("data_version"),
            loaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            drift=drift,
        )
        with self._lock:
            self._current = loaded
            self.last_error = None
        log.info(
            "model_loaded",
            model_name=s.model_name,
            model_version=loaded.version,
            alias=s.model_alias,
            run_id=loaded.run_id,
            data_version=loaded.data_version,
        )
        return loaded

    def _drift_monitor(self, run_id: str | None, version: str) -> DriftMonitor | None:
        """Download the training profile of the champion's run (artifact of ml/train.py)."""
        import mlflow.artifacts

        if not run_id:
            return None
        try:
            profile = mlflow.artifacts.load_dict(f"runs:/{run_id}/{PROFILE_ARTIFACT}")
        except Exception as exc:  # no profile on this run, or artifact store unreachable
            log.warning(
                "drift_profile_missing",
                model_version=version,
                run_id=run_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        s = self.settings
        log.info(
            "drift_profile_loaded",
            model_version=version,
            run_id=run_id,
            data_version=profile.get("source", {}).get("data_version"),
            window_size=s.drift_window_size,
            recompute_every=s.drift_recompute_every,
        )
        return DriftMonitor(profile, version, s.drift_window_size, s.drift_recompute_every)

    def try_load(self) -> bool:
        try:
            self.load()
            return True
        except Exception as exc:  # registry down, alias missing, bad model...
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.error(
                "model_load_failed", model_name=self.settings.model_name, error=self.last_error
            )
            return False

    async def load_with_retry(self) -> None:
        """Background task: retry with exponential backoff until a model is loaded."""
        delay = self.settings.retry_initial_delay
        while not await asyncio.to_thread(self.try_load):
            log.info("model_load_retry", delay_s=delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.settings.retry_max_delay)

    # --- serving ----------------------------------------------------------------------

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def is_ready(self) -> bool:
        """Ready only if a model is loaded AND an internal test prediction succeeds."""
        current = self._current
        if current is None:
            return False
        try:
            _spam_probabilities(current.model, PROBE_TEXTS[:1])
        except Exception as exc:
            self.last_error = f"self-check failed: {exc}"
            return False
        return True

    def predict(self, text: str) -> tuple[str, float, LoadedModel]:
        current = self._current
        if current is None:
            raise ModelNotReadyError(self.last_error or "model not loaded yet")
        prob = _spam_probabilities(current.model, [text])[0]
        label = SPAM if prob >= 0.5 else "ham"
        return label, prob, current
