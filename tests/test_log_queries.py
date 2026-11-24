"""The jq queries documented in docs/MONITORING.md, run on real logs of the app.

The logs come from the application itself (two model versions, as after a rollback,
plus a 503 while no model is loaded). Skipped when jq is not installed (it is on the
GitHub Actions runners, so CI always checks that the documentation is right).
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from mlflow import MlflowClient

from app.logging_config import configure_logging
from tests.conftest import UNREACHABLE_URI

DOC = Path(__file__).resolve().parents[1] / "docs" / "MONITORING.md"

ERRORS = """jq -R -c 'fromjson? | select(.level == "error")'"""
ONE_REQUEST = """jq -R -c 'fromjson? | select(.request_id == "<id>")'"""
PER_VERSION = (
    """jq -R -s -c '[split("\\n")[] | fromjson? | select(.event == "prediction_made")] """
    """| group_by(.model_version) | map({model_version: .[0].model_version, """
    """predictions: length})'"""
)
PER_VERSION_LABEL = (
    """jq -R -s -c '[split("\\n")[] | fromjson? | select(.event == "prediction_made")] """
    """| group_by([.model_version, .prediction]) | map({model_version: .[0].model_version, """
    """prediction: .[0].prediction, n: length})'"""
)
SLOW = (
    """jq -R -c 'fromjson? | select(.event == "request_completed" and .duration_ms > 500) """
    """| {timestamp, request_id, path, status, duration_ms}'"""
)
SERVER_ERRORS_BY_ROUTE = (
    """jq -R -s -c '[split("\\n")[] | fromjson? | select(.event == "request_completed" """
    """and .status >= 500)] | group_by(.path) | map({path: .[0].path, errors: length})'"""
)
MODEL_TIMELINE = (
    """jq -R -c 'fromjson? | select(.event == "model_loaded") """
    """| {timestamp, model_version}'"""
)

ALL_QUERIES = [
    ERRORS,
    ONE_REQUEST,
    PER_VERSION,
    PER_VERSION_LABEL,
    SLOW,
    SERVER_ERRORS_BY_ROUTE,
    MODEL_TIMELINE,
]


def test_documented_queries_are_the_tested_ones():
    doc = DOC.read_text(encoding="utf-8")
    for query in ALL_QUERIES:
        assert query in doc, f"query missing from docs/MONITORING.md: {query}"


def jq(query: str, text: str, request_id: str = "") -> list:
    """Run a documented query (``jq ... '<filter>'``) on ``text``; one JSON value per line."""
    program = query.split("'", 1)[1].rsplit("'", 1)[0].replace("<id>", request_id)
    flags = query.split("'", 1)[0].split()[1:]
    out = subprocess.run(
        ["jq", *flags, program], input=text, capture_output=True, text=True, check=True
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def app_logs(tmp_path_factory) -> str:
    """Logs of a real session: v1 serves 3 predictions, the champion moves to v2 (as a
    rollback or a promotion would), v2 serves 2, and a service without a model answers
    a 503. A non-JSON line is mixed in, as a forgotten log prefix would."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_registry, wait_for

    registry = make_registry(tmp_path_factory.mktemp("logq"))
    buffer = io.StringIO()
    configure_logging(level="INFO", stream=buffer)
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("MLFLOW_TRACKING_URI", registry["tracking_uri"])
        mp.setenv("MODEL_LOAD_RETRY_INITIAL_DELAY", "0.2")
        mp.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        mp.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")

        def serve(texts: list[str], prefix: str) -> None:
            with TestClient(create_app()) as client:
                assert wait_for(lambda: client.get("/ready").status_code == 200)
                for i, text in enumerate(texts):
                    headers = {"X-Request-ID": f"{prefix}-{i}"}
                    assert (
                        client.post("/predict", json={"text": text}, headers=headers).status_code
                        == 200
                    )

        serve(["see you at lunch", "WIN a free prize now", "ok thanks"], "v1")
        client = MlflowClient(registry["tracking_uri"])
        newer = registry["newer_version"]
        client.set_model_version_tag("sms-spam", newer, "status", "approved")
        client.set_registered_model_alias("sms-spam", "champion", newer)
        serve(["call me later", "URGENT claim your cash"], "v2")

        mp.setenv("MLFLOW_TRACKING_URI", UNREACHABLE_URI)
        with TestClient(create_app()) as down:
            resp = down.post("/predict", json={"text": "hi"}, headers={"X-Request-ID": "down-0"})
            assert resp.status_code == 503
    finally:
        mp.undo()
        configure_logging()
    return buffer.getvalue() + "not json: a forgotten prefix\n"


needs_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")


@needs_jq
def test_errors_only(app_logs):
    errors = jq(ERRORS, app_logs)
    assert errors and all(line["level"] == "error" for line in errors)
    assert any(line.get("request_id") == "down-0" and line["status"] == 503 for line in errors)


@needs_jq
def test_follow_one_request(app_logs):
    lines = jq(ONE_REQUEST, app_logs, request_id="v1-1")
    assert [line["event"] for line in lines] == ["prediction_made", "request_completed"]
    assert {line["request_id"] for line in lines} == {"v1-1"}


@needs_jq
def test_predictions_per_model_version(app_logs):
    (counts,) = jq(PER_VERSION, app_logs)
    # Fresh test registry: version 1 served 3 predictions, then version 2 served 2.
    assert {row["model_version"]: row["predictions"] for row in counts} == {"1": 3, "2": 2}


@needs_jq
def test_other_queries(app_logs):
    (per_label,) = jq(PER_VERSION_LABEL, app_logs)
    assert sum(row["n"] for row in per_label) == 5
    assert all(row["duration_ms"] > 500 for row in jq(SLOW, app_logs))
    (by_route,) = jq(SERVER_ERRORS_BY_ROUTE, app_logs)
    assert {"path": "/predict", "errors": 1} in by_route
    versions = [row["model_version"] for row in jq(MODEL_TIMELINE, app_logs)]
    assert versions[:2] == ["1", "2"]


def test_app_logs_contain_what_the_queries_need(app_logs):
    """Without jq: the same expectations, checked in Python on the same logs."""
    lines = []
    for raw in app_logs.splitlines():
        try:
            lines.append(json.loads(raw))
        except ValueError:
            continue  # what fromjson? skips
    made = [line for line in lines if line["event"] == "prediction_made"]
    assert sorted(line["model_version"] for line in made) == ["1", "1", "1", "2", "2"]
    one = [line["event"] for line in lines if line["request_id"] == "v1-1"]
    assert one == ["prediction_made", "request_completed"]
    errors = [line for line in lines if line["level"] == "error"]
    assert any(line.get("request_id") == "down-0" and line["status"] == 503 for line in errors)
    loaded = [line["model_version"] for line in lines if line["event"] == "model_loaded"]
    assert loaded[:2] == ["1", "2"]
