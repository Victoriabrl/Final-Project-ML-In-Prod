"""Shared fixtures: a temporary local MLflow registry (sqlite + artifacts in tmp) and
app clients configured through environment variables. No network access."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from scripts.make_test_registry import build_registry

UNREACHABLE_URI = "http://127.0.0.1:9"  # nothing listens on the discard port


def make_registry(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return build_registry(
        f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        artifact_location=(root / "artifacts").as_uri(),
    )


@pytest.fixture(scope="session")
def registry(tmp_path_factory) -> dict:
    """Shared read-only registry: v1 = champion (approved), v2 = newer candidate."""
    return make_registry(tmp_path_factory.mktemp("registry"))


@pytest.fixture
def fresh_registry(tmp_path) -> dict:
    """Private registry for tests that move aliases or change tags."""
    return make_registry(tmp_path / "registry")


def wait_for(predicate, timeout: float = 30, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def make_client(monkeypatch):
    """Factory: start the app (lifespan included) against a given registry."""
    clients: list[TestClient] = []

    def _make(tracking_uri: str, alias: str = "champion", wait_ready: bool = True) -> TestClient:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("MODEL_NAME", "sms-spam")
        monkeypatch.setenv("MODEL_ALIAS", alias)
        monkeypatch.setenv("MODEL_LOAD_RETRY_INITIAL_DELAY", "0.2")
        monkeypatch.setenv("MODEL_LOAD_RETRY_MAX_DELAY", "0.5")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
        client = TestClient(create_app())
        client.__enter__()  # runs the lifespan (starts background loading)
        clients.append(client)
        if wait_ready:
            assert wait_for(lambda: client.get("/ready").status_code == 200), "app never ready"
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(registry, make_client) -> TestClient:
    return make_client(registry["tracking_uri"])
