"""Shared fixtures: a temporary local MLflow registry (sqlite + artifacts in tmp) and
app clients configured through environment variables. No network access."""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from app.main import create_app
from ml.ct_pipeline import CT_EXPERIMENT
from ml.data_utils import file_entry, write_csv, write_json
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


# --- CT pipeline: tiny hand-made releases and an empty local registry --------------

SPAM_WORDS = [
    "win a free cash prize",
    "claim your reward now",
    "urgent prize call",
    "free voucher txt",
]


def frame(n_ham: int, n_spam: int, tag: str) -> pd.DataFrame:
    ham = [f"see you at lunch {tag} day {i}, ok thanks" for i in range(n_ham)]
    spam = [f"{SPAM_WORDS[i % 4]} {tag} code {i} call 0800" for i in range(n_spam)]
    return pd.DataFrame({"label": ["ham"] * n_ham + ["spam"] * n_spam, "text": ham + spam})


def write_release(root: Path, version: str, train: pd.DataFrame, holdout: pd.DataFrame) -> None:
    out = root / version
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in (("train.csv", train), ("holdout.csv", holdout)):
        write_csv(df, out / name)
        files[name] = file_entry(df, out / name)
    write_json({"version": version, "seed": 42, "files": files}, out / "manifest.json")


@pytest.fixture
def releases(tmp_path) -> Path:
    """v1 (small) and v2 (bigger) releases sharing the same held-out set."""
    root = tmp_path / "releases"
    holdout = frame(12, 3, "holdout")
    write_release(root, "v1", frame(20, 4, "train"), holdout)
    write_release(root, "v2", frame(40, 8, "train"), holdout)
    return root


@pytest.fixture
def ct_registry(tmp_path):
    """Empty sqlite registry; artifacts kept under tmp_path (not ./mlruns)."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    client.create_experiment(CT_EXPERIMENT, artifact_location=(tmp_path / "art").as_uri())
    yield client
    mlflow.set_tracking_uri(previous)
