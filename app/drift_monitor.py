"""Live input drift: recent predictions compared with the champion's training profile.

- The profile (``monitoring/training-profile.json``) is downloaded from the champion's
  training run when the model is loaded (``ModelLoader``); without one, drift is simply
  not computed (the service stays ready: drift is a signal, not a dependency).
- A bounded window keeps the **features** of the last ``window_size`` predictions (500 by
  default), never the text: ``deque(maxlen=...)``, so memory is capped.
- Every ``recompute_every`` predictions (100), once the window is full, the PSI of each
  feature is recomputed with ``app.drift.feature_psi``: the same bins, PSI and 0.20
  threshold as the CT pipeline's ``drift.json``, so the values mean the same thing.
  Waiting for a full window avoids false alarms: on 100 messages, sampling noise alone
  can exceed 0.20 (see ml/README_experiments.md).
- The latest values feed the ``feature_drift{feature}`` gauge and a ``drift_computed``
  log line (level warning when a feature crosses the threshold: investigate, do not
  retrain automatically).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping

from app.drift import DRIFT_THRESHOLD, SCORE_FEATURE, TEXT_FEATURES, feature_psi, text_features
from app.logging_config import get_logger

log = get_logger("app.drift")


class DriftMonitor:
    def __init__(
        self,
        profile: Mapping,
        model_version: str,
        window_size: int = 500,
        recompute_every: int = 100,
    ) -> None:
        if window_size < 1 or recompute_every < 1:
            raise ValueError("window_size and recompute_every must be positive")
        self.profile = profile
        self.model_version = model_version
        self.window_size = window_size
        self.recompute_every = recompute_every
        self.threshold = float(profile.get("threshold", DRIFT_THRESHOLD))
        self.window: deque[dict[str, float]] = deque(maxlen=window_size)
        self.observed = 0
        self.computations = 0
        self.latest: dict[str, float] = {}
        self._lock = threading.Lock()

    def observe(self, text: str, score: float) -> dict[str, float] | None:
        """Record one prediction; returns the new PSI values when they were recomputed."""
        row = text_features(text)
        row[SCORE_FEATURE] = float(score)
        with self._lock:
            self.window.append(row)
            self.observed += 1
            due = self.observed % self.recompute_every == 0
            if not due or len(self.window) < self.window_size:
                return None
            columns = {
                name: [r[name] for r in self.window] for name in (*TEXT_FEATURES, SCORE_FEATURE)
            }
            self.latest = feature_psi(self.profile, columns)
            self.computations += 1
            values = dict(self.latest)
        drifted = sorted(name for name, v in values.items() if v >= self.threshold)
        fields = {
            "model_version": self.model_version,
            "window": len(self.window),
            "psi": {k: round(v, 4) for k, v in values.items()},
            "max_psi": round(max(values.values()), 4) if values else 0.0,
            "threshold": self.threshold,
            "drifted": drifted,
        }
        if drifted:
            log.warning("drift_computed", **fields)
        else:
            log.info("drift_computed", **fields)
        return values
