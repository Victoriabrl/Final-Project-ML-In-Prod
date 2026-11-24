"""Structured logging: JSON lines, X-Request-ID, request/prediction events, no leaks."""

from __future__ import annotations

import io
import json
import logging
import os

import pytest

from app.logging_config import configure_logging, get_logger
from tests.conftest import UNREACHABLE_URI, wait_for

SMS = "UNIQUE-SMS-ZQX please call me back about lunch"


@pytest.fixture
def logs():
    """Send the app's logs to a buffer; returns a function parsing every line as JSON."""
    buffer = io.StringIO()
    configure_logging(level="INFO", stream=buffer)

    def lines() -> list[dict]:
        # json.loads fails the test if any line is not JSON.
        return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]

    lines.raw = buffer.getvalue
    yield lines
    configure_logging()


def events(lines: list[dict], name: str) -> list[dict]:
    return [line for line in lines if line.get("event") == name]


def test_every_line_is_json_and_has_a_request_id_key(logs, registry, make_client):
    client = make_client(registry["tracking_uri"])
    client.post("/predict", json={"text": SMS})
    client.get("/model-info")
    lines = logs()
    assert lines, "no log line captured"
    assert all("request_id" in line and "timestamp" in line and "level" in line for line in lines)


def test_one_request_id_follows_the_request(logs, registry, make_client):
    client = make_client(registry["tracking_uri"])
    resp = client.post("/predict", json={"text": SMS}, headers={"X-Request-ID": "demo-123"})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "demo-123"

    mine = [line for line in logs() if line["request_id"] == "demo-123"]
    assert {line["event"] for line in mine} >= {"request_completed", "prediction_made"}
    done = events(mine, "request_completed")[0]
    assert done["method"] == "POST" and done["path"] == "/predict" and done["status"] == 200
    assert done["duration_ms"] >= 0 and done["level"] == "info"


def test_request_id_is_generated_or_replaced_when_invalid(logs, client):
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 32
    other = client.get("/health").headers["X-Request-ID"]
    assert other != generated
    # A header that could inject content into the logs is not reused.
    injected = client.get("/health", headers={"X-Request-ID": "bad id\nlevel=error"})
    assert injected.headers["X-Request-ID"] != "bad id\nlevel=error"
    long_id = client.get("/health", headers={"X-Request-ID": "a" * 200})
    assert long_id.headers["X-Request-ID"] != "a" * 200


def test_prediction_made_fields(logs, registry, client):
    client.post("/predict", json={"text": SMS}, headers={"X-Request-ID": "pred-1"})
    made = [e for e in events(logs(), "prediction_made") if e["request_id"] == "pred-1"][0]
    assert made["model_version"] == registry["champion_version"]
    assert made["prediction"] in {"spam", "ham"}
    assert 0.0 <= made["score"] <= 1.0
    assert made["latency_ms"] >= 0
    assert made["n_chars"] == len(SMS) and made["n_words"] == len(SMS.split())


def test_model_loaded_event(logs, registry, make_client):
    make_client(registry["tracking_uri"])
    loaded = events(logs(), "model_loaded")
    assert loaded, "model_loaded not logged"
    assert loaded[-1]["model_name"] == "sms-spam"
    assert loaded[-1]["model_version"] == registry["champion_version"]
    assert loaded[-1]["request_id"] is None  # startup, outside any request


def test_server_errors_are_logged_at_error_level(logs, make_client):
    client = make_client(UNREACHABLE_URI, wait_ready=False)
    resp = client.post("/predict", json={"text": SMS}, headers={"X-Request-ID": "down-1"})
    assert resp.status_code == 503
    done = [e for e in events(logs(), "request_completed") if e["request_id"] == "down-1"][0]
    assert done["status"] == 503 and done["level"] == "error"
    assert wait_for(lambda: events(logs(), "model_load_failed"))
    assert events(logs(), "model_load_failed")[0]["level"] == "error"


def test_no_sms_text_and_no_credentials_in_logs(logs, registry, make_client, monkeypatch):
    secret = "s3cr3t-dagshub-token-xyz"
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", secret)
    client = make_client(registry["tracking_uri"])
    client.post("/predict", json={"text": SMS})
    client.post("/predict", json={"text": "a" * 1001})  # rejected input
    # Even if a library writes a secret or URL credentials in a message, it is masked.
    get_logger("test").error("boom", detail=f"token={secret} https://user:pw123@host/x")
    raw = logs.raw()
    assert "UNIQUE-SMS-ZQX" not in raw and "a" * 1001 not in raw
    assert secret not in raw and "pw123" not in raw
    assert "***" in raw


def test_route_template_is_logged_not_the_raw_url(logs, client):
    client.get("/model-info?debug=1", headers={"X-Request-ID": "tpl-1"})
    client.get("/does-not-exist", headers={"X-Request-ID": "tpl-2"})
    done = {e["request_id"]: e for e in events(logs(), "request_completed")}
    assert done["tpl-1"]["path"] == "/model-info"
    assert done["tpl-2"]["path"] == "<unmatched>" and done["tpl-2"]["status"] == 404


def test_only_json_from_libraries_and_no_access_log():
    buffer = io.StringIO()
    configure_logging(level="INFO", stream=buffer)
    try:
        logging.getLogger("uvicorn.error").info("Started server process [%d]", 42)
        logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /predict" 200')
        logging.getLogger("mlflow.store").warning("registry slow")
    finally:
        configure_logging()
    lines = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert [line["event"] for line in lines] == ["Started server process [42]", "registry slow"]
    assert os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] == "false"


def test_log_level_from_configuration():
    buffer = io.StringIO()
    configure_logging(level="WARNING", stream=buffer)
    try:
        get_logger("test").info("hidden_event")
        get_logger("test").warning("shown_event")
    finally:
        configure_logging()
    out = buffer.getvalue()
    assert "hidden_event" not in out and "shown_event" in out
