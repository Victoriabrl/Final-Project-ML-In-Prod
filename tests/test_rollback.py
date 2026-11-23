"""Rollback: target resolution, approval check, alias move and MLflow logging."""

from __future__ import annotations

import mlflow
import pytest
from mlflow import MlflowClient

from ml.registry_utils import promote
from ml.rollback import OPERATIONS_EXPERIMENT, RollbackError, main, rollback
from tests.conftest import make_registry

MODEL = "sms-spam"


@pytest.fixture
def reg(tmp_path):
    """v1 approved champion, v2 newer candidate; global tracking URI on this registry."""
    info = make_registry(tmp_path / "registry")
    client = MlflowClient(info["tracking_uri"])
    client.create_experiment(OPERATIONS_EXPERIMENT, artifact_location=(tmp_path / "art").as_uri())
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(info["tracking_uri"])
    yield client
    mlflow.set_tracking_uri(previous)


def champion(client: MlflowClient) -> str:
    return str(client.get_model_version_by_alias(MODEL, "champion").version)


def promote_v2(client: MlflowClient) -> None:
    """Simulate a CT promotion of v2 (approved, alias, previous_champion=1)."""
    promote(client, MODEL, "2")
    assert champion(client) == "2"


def test_default_target_is_previous_champion(reg):
    promote_v2(reg)
    result = rollback(MODEL, client=reg)
    assert (result.from_version, result.to_version, result.changed) == ("2", "1", True)
    assert champion(reg) == "1"


def test_explicit_approved_target(reg):
    promote_v2(reg)
    rollback(MODEL, "1", client=reg)
    result = rollback(MODEL, "2", client=reg)  # roll forward again to the best model
    assert result.to_version == "2" and champion(reg) == "2"


def test_unapproved_target_is_refused_and_nothing_changes(reg):
    with pytest.raises(RollbackError, match="status 'candidate'"):
        rollback(MODEL, "2", client=reg)
    assert champion(reg) == "1"


def test_unknown_version_is_refused(reg):
    with pytest.raises(RollbackError, match="does not exist"):
        rollback(MODEL, "99", client=reg)
    assert champion(reg) == "1"


def test_no_previous_champion_needs_explicit_version(reg):
    with pytest.raises(RollbackError, match="previous_champion"):
        rollback(MODEL, client=reg)


def test_target_already_champion_is_a_noop(reg):
    result = rollback(MODEL, "1", client=reg)
    assert result.changed is False and champion(reg) == "1"


def test_rollback_is_logged_in_mlflow(reg):
    promote_v2(reg)
    result = rollback(MODEL, reason="demo", client=reg)
    run = reg.get_run(result.run_id)
    assert run.info.run_name == "rollback"
    t = run.data.tags
    assert t["champion_version_before"] == "2" and t["champion_version_after"] == "1"
    assert t["reason"] == "demo" and t["changed"] == "true" and t["git_sha"]


def test_cli_writes_github_outputs(reg, monkeypatch, tmp_path):
    promote_v2(reg)
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["--model-name", MODEL]) == 0
    assert output.read_text() == "from_version=2\nto_version=1\nchanged=true\n"


def test_cli_refusal_exits_non_zero(reg, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert main(["--model-name", MODEL, "--version", "2"]) == 1
    assert champion(reg) == "1"
