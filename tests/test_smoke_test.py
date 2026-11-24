"""scripts/smoke_test.py waits for the new instance after a Render redeploy."""

from __future__ import annotations

import itertools

import pytest

from scripts import smoke_test


def fake_server(monkeypatch, versions):
    """/ready answers with the next model version of `versions` on each call."""
    # The last version repeats forever (an instance that never changes).
    calls = itertools.chain(versions, itertools.repeat(versions[-1]))

    def request(method, url, body=None, timeout=30):
        assert url.endswith("/ready")
        return 200, {"status": "ready", "model_version": next(calls), "detail": None}

    monkeypatch.setattr(smoke_test, "request", request)
    monkeypatch.setattr(smoke_test.time, "sleep", lambda _: None)


def test_wait_ready_waits_for_expected_model_version(monkeypatch):
    # Old instance (version 1) keeps serving while the new one (version 2) starts.
    fake_server(monkeypatch, ["1", "1", "2"])
    body = smoke_test.wait_ready("http://prod", timeout=60, expected_version="2", interval=0)
    assert body["model_version"] == "2"


def test_wait_ready_without_expected_version_takes_first_ready(monkeypatch):
    fake_server(monkeypatch, ["1"])
    assert smoke_test.wait_ready("http://prod", timeout=60, interval=0)["model_version"] == "1"


def test_wait_ready_times_out_on_wrong_version(monkeypatch):
    fake_server(monkeypatch, ["1"])
    with pytest.raises(AssertionError, match="model version '1', expected '2'"):
        smoke_test.wait_ready("http://prod", timeout=0.05, expected_version="2", interval=0)
