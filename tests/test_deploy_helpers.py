"""Helpers used by the CD workflows: /health commit, champion resolution."""

from __future__ import annotations

import pytest
from mlflow import MlflowClient

from scripts.resolve_champion import main as resolve_main
from scripts.resolve_champion import resolve


def test_health_reports_image_commit(registry, make_client, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abc1234")
    client = make_client(registry["tracking_uri"], wait_ready=False)
    assert client.get("/health").json() == {"status": "ok", "git_sha": "abc1234"}


def test_health_without_commit(registry, make_client, monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    client = make_client(registry["tracking_uri"], wait_ready=False)
    assert client.get("/health").json() == {"status": "ok", "git_sha": None}


def test_resolve_champion_gives_concrete_version(registry):
    client = MlflowClient(registry["tracking_uri"])
    assert resolve("sms-spam", "champion", client=client) == registry["champion_version"]


def test_resolve_missing_alias_fails(registry):
    client = MlflowClient(registry["tracking_uri"])
    with pytest.raises(LookupError, match="no 'nope' alias"):
        resolve("sms-spam", "nope", client=client)


def test_resolve_unapproved_champion_fails(fresh_registry):
    client = MlflowClient(fresh_registry["tracking_uri"])
    client.set_model_version_tag("sms-spam", "1", "status", "candidate")
    with pytest.raises(LookupError, match="candidate"):
        resolve("sms-spam", "champion", client=client)


def test_resolve_cli_writes_github_output(registry, monkeypatch, tmp_path, capsys):
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    argv = ["--model-name", "sms-spam", "--tracking-uri", registry["tracking_uri"]]
    assert resolve_main(argv) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert output.read_text() == "found=true\nversion=1\n"


def test_resolve_cli_allow_missing_for_bootstrap(registry, monkeypatch, tmp_path):
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    argv = ["--alias", "nope", "--allow-missing", "--tracking-uri", registry["tracking_uri"]]
    assert resolve_main(argv) == 0
    assert output.read_text() == "found=false\n"


def test_resolve_cli_fails_without_alias(registry, monkeypatch, tmp_path):
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert resolve_main(["--alias", "nope", "--tracking-uri", registry["tracking_uri"]]) == 1
    assert not output.exists()  # nothing written for the workflow on failure
