"""CT pipeline on tiny hand-made releases and a temporary local registry (no network)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from ml.ct_pipeline import CT_EXPERIMENT, CTError, run_ct
from ml.data_utils import file_entry, write_csv, write_json

MODEL = "sms-spam"
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


@pytest.fixture
def lenient_config(tmp_path) -> Path:
    """Same rule but min_improvement 0, so an equally good candidate is promoted."""
    path = tmp_path / "promotion.yaml"
    path.write_text(
        "metric: f1_spam\ndirection: maximize\nquality_floor: 0.85\nmin_improvement: 0.0\n"
        "retrain_when: [new_data_version, champion_below_floor]\n",
        encoding="utf-8",
    )
    return path


class FakeDeployer:
    """Records deploys; production 'serves' whatever version ``serves`` returns."""

    def __init__(self, serves=lambda expected: True):
        self.serves = serves
        self.deploys = 0
        self.checked: list[str] = []

    def deploy(self) -> None:
        self.deploys += 1

    def verify(self, expected_version: str) -> bool:
        self.checked.append(expected_version)
        return self.serves(expected_version)


def champion(client: MlflowClient) -> str:
    return str(client.get_model_version_by_alias(MODEL, "champion").version)


def tags(client: MlflowClient, version: str) -> dict:
    return client.get_model_version(MODEL, version).tags


def test_bootstrap_trains_baseline_and_promotes_it(ct_registry, releases):
    result = run_ct("v1", client=ct_registry, data_root=releases)
    assert result.decision == "promoted"
    assert result.retrain_reasons == ["no_champion"]
    assert result.champion_before is None and result.champion_after == "1"
    assert champion(ct_registry) == "1"
    assert tags(ct_registry, "1")["status"] == "approved"
    child = ct_registry.get_run(ct_registry.get_model_version(MODEL, "1").run_id)
    assert child.data.tags["model_family"] == "baseline"  # bootstrap uses the baseline


def test_same_data_version_keeps_champion_without_training(ct_registry, releases):
    run_ct("v1", client=ct_registry, data_root=releases)
    result = run_ct("v1", client=ct_registry, data_root=releases)
    assert result.decision == "kept"
    assert "no reason to retrain" in result.reason
    assert result.candidate_version is None
    assert len(ct_registry.search_model_versions(f"name='{MODEL}'")) == 1


def test_forced_candidate_without_improvement_is_rejected(ct_registry, releases):
    run_ct("v1", client=ct_registry, data_root=releases)
    result = run_ct("v1", force_train=True, client=ct_registry, data_root=releases)
    assert result.decision == "rejected"
    assert result.candidate_version == "2"
    assert champion(ct_registry) == "1"  # champion kept
    assert tags(ct_registry, "2")["status"] == "rejected"
    assert "improvement" in tags(ct_registry, "2")["status_reason"]


def test_new_data_version_retrains_and_promotes(ct_registry, releases, lenient_config):
    run_ct("v1", client=ct_registry, data_root=releases)
    result = run_ct("v2", client=ct_registry, data_root=releases, promotion_config=lenient_config)
    assert result.retrain_reasons == ["new_data_version"]
    assert result.decision == "promoted"
    assert champion(ct_registry) == "2"
    assert tags(ct_registry, "2")["previous_champion"] == "1"
    assert tags(ct_registry, "2")["data_version"] == "v2"


def test_invalid_release_changes_nothing(ct_registry, releases):
    (releases / "v1" / "train.csv").write_text("label,text\nham,tampered\n", encoding="utf-8")
    with pytest.raises(CTError, match="data contract failed"):
        run_ct("v1", client=ct_registry, data_root=releases)
    assert ct_registry.search_registered_models() == []


def test_promotion_deploys_and_checks_production(ct_registry, releases):
    deployer = FakeDeployer()
    result = run_ct("v1", deploy=True, deployer=deployer, client=ct_registry, data_root=releases)
    assert result.deployed is True
    assert deployer.deploys == 1 and deployer.checked == ["1"]


def test_failed_production_check_restores_previous_champion(ct_registry, releases, lenient_config):
    run_ct("v1", client=ct_registry, data_root=releases)
    deployer = FakeDeployer(serves=lambda expected: expected == "1")  # new version never live
    with pytest.raises(CTError, match="alias restored to 1"):
        run_ct(
            "v2",
            deploy=True,
            deployer=deployer,
            client=ct_registry,
            data_root=releases,
            promotion_config=lenient_config,
        )
    assert champion(ct_registry) == "1"
    assert tags(ct_registry, "2")["status"] == "rejected"
    assert deployer.checked == ["2", "1"] and deployer.deploys == 2  # redeployed + verified


def test_staging_model_is_never_deployed(ct_registry, releases):
    deployer = FakeDeployer()
    result = run_ct(
        "v1",
        model_name="sms-spam-staging",
        deploy=True,
        deployer=deployer,
        client=ct_registry,
        data_root=releases,
    )
    assert result.decision == "promoted" and result.deployed is False
    assert deployer.deploys == 0


def test_ct_decision_run_is_logged(ct_registry, releases):
    result = run_ct("v1", client=ct_registry, data_root=releases)
    run = ct_registry.get_run(result.run_id)
    assert run.info.run_name == "ct-decision"
    t = run.data.tags
    assert t["decision"] == "promoted"
    assert t["data_version"] == "v1"
    assert t["git_sha"]
    assert t["champion_version_before"] == "none" and t["champion_version_after"] == "1"
    assert run.data.params["quality_floor"] == "0.85"
    assert "candidate_f1_spam" in run.data.metrics
    # The training run is nested under the decision run.
    child = ct_registry.get_run(ct_registry.get_model_version(MODEL, "1").run_id)
    assert child.data.tags["mlflow.parentRunId"] == result.run_id
