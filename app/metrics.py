"""Prometheus metrics exposed on GET /metrics (prometheus_client, text format).

The 5 metrics of the monitoring brief, with low-cardinality labels only (route template,
method, status, predicted class, model version, feature name: never a request id or
an input value):

- ``http_requests_total{method, path, status}``      counter, every HTTP request
- ``http_request_duration_seconds{path}``           histogram, request latency
- ``predictions_total{prediction, model_version}``  counter, spam / ham per version
- ``model_info{model_version}``                     gauge = 1, only once a model is loaded
- ``feature_drift{feature}``                        gauge, PSI per feature (set by the
                                                    drift monitor, step 5)

Each app gets its own registry (tests create several apps in one process). Counters
start at 0 when the process starts: dashboards use rate() / increase(), never raw values.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest

__all__ = ["CONTENT_TYPE_LATEST", "Metrics"]


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "http_requests_total",
            "HTTP requests handled, by method, route template and status code.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds, by route template.",
            ["path"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "predictions_total",
            "Predictions served, by predicted class and concrete model version.",
            ["prediction", "model_version"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            "model_info",
            "1 for the concrete model version currently served (absent before loading).",
            ["model_version"],
            registry=self.registry,
        )
        self.feature_drift = Gauge(
            "feature_drift",
            "PSI between recent inputs and the champion's training profile, by feature.",
            ["feature"],
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        self.http_requests.labels(method=method, path=path, status=str(status)).inc()
        self.http_duration.labels(path=path).observe(seconds)

    def observe_prediction(self, prediction: str, model_version: str) -> None:
        self.predictions.labels(prediction=prediction, model_version=model_version).inc()

    def set_model_version(self, model_version: str | None) -> None:
        """model_info has exactly one series (the served version), or none before loading."""
        self.model_info.clear()
        if model_version is not None:
            self.model_info.labels(model_version=model_version).set(1)

    def generate_latest(self) -> bytes:
        return _generate_latest(self.registry)
