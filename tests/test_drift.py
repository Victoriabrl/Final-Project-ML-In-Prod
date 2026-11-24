"""Drift module (app/drift.py), training profile, CT drift.json and profile backfill."""

from __future__ import annotations

import json

import mlflow
import mlflow.artifacts
import pytest
from mlflow.exceptions import MlflowException

from app.drift import (
    DRIFT_THRESHOLD,
    FEATURES,
    PROFILE_ARTIFACT,
    build_profile,
    drift_report,
    psi,
    text_features,
)
from ml.backfill_profile import backfill, has_profile
from ml.ct_pipeline import run_ct
from tests.conftest import frame, write_release
from tests.test_ct_pipeline import MODEL

NORMAL = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Ok, I'll call you when I get home",
    "Can you pick up some milk on the way back",
    "Sorry, running 10 min late, see you soon",
    "Happy birthday! Have a great day",
    "WINNER!! Claim your free prize, call 09061701461 now",
] * 50
# A spam campaign: shouting, numbers and links everywhere.
CAMPAIGN = [
    "URGENT!! YOU WON 2000 GBP. CALL 09061701461 OR VISIT WWW.CLAIM-NOW.COM CODE 81010",
    "FREE ENTRY 2 WIN 1000 CASH! TXT WIN TO 87121 NOW 18+ T&C WWW.WIN.COM",
] * 150


def scores_like(texts: list[str]) -> list[float]:
    return [0.95 if "!!" in t or "FREE" in t else 0.05 for t in texts]


@pytest.fixture(scope="module")
def profile() -> dict:
    return build_profile(NORMAL, scores_like(NORMAL), source={"data_version": "test"})


def test_text_features_are_numbers_only():
    f = text_features("WIN 1000 cash! www.win.com")
    assert set(f) == {"length_chars", "digit_ratio", "upper_ratio", "has_url_or_phone"}
    assert f["length_chars"] == 26.0
    assert f["digit_ratio"] == pytest.approx(4 / 26)
    assert f["upper_ratio"] == pytest.approx(3 / 16)
    assert f["has_url_or_phone"] == 1.0
    assert text_features("")["digit_ratio"] == 0.0  # no division by zero
    assert text_features("see you later")["has_url_or_phone"] == 0.0


def test_profile_is_json_and_keeps_no_text(profile):
    dumped = json.dumps(profile)
    assert set(profile["features"]) == set(FEATURES)
    assert profile["threshold"] == DRIFT_THRESHOLD
    assert "lunch" not in dumped and "WINNER" not in dumped
    for ref in profile["features"].values():
        assert sum(ref["proportions"]) == pytest.approx(1.0)
        assert len(ref["proportions"]) == len(ref["edges"]) + 1


def test_same_distribution_is_below_threshold(profile):
    report = drift_report(profile, NORMAL, scores_like(NORMAL))
    assert report["max_psi"] < DRIFT_THRESHOLD
    assert report["drifted"] == []
    assert set(report["psi"]) == set(FEATURES)


def test_spam_campaign_is_above_threshold(profile):
    report = drift_report(profile, CAMPAIGN, scores_like(CAMPAIGN))
    assert report["max_psi"] >= DRIFT_THRESHOLD
    for name in ("upper_ratio", "digit_ratio", "spam_probability"):
        assert name in report["drifted"]


def test_rare_binary_feature_can_drift():
    # 10% of reference messages have a link or number; a batch where 80% do must drift.
    reference = ["see you soon"] * 90 + ["call 09061701461"] * 10
    batch = ["see you soon"] * 20 + ["call 09061701461"] * 80
    prof = build_profile(reference, [0.1] * 100)
    assert prof["features"]["has_url_or_phone"]["edges"] == [0.5]
    report = drift_report(prof, batch, [0.1] * 100)
    assert "has_url_or_phone" in report["drifted"]


def test_psi_handles_empty_bins():
    assert psi([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]) == 0.0
    value = psi([1.0, 0.0], [0.0, 1.0])
    assert value > DRIFT_THRESHOLD and value != float("inf")


def test_training_logs_profile_and_ct_logs_drift(ct_registry, releases):
    run_ct("v1", client=ct_registry, data_root=releases)
    run_id = ct_registry.get_model_version(MODEL, "1").run_id
    profile = mlflow.artifacts.load_dict(f"runs:/{run_id}/{PROFILE_ARTIFACT}")
    assert profile["source"]["data_version"] == "v1"
    assert set(profile["features"]) == set(FEATURES)

    # Second run: the champion exists and has a profile, so drift.json is logged.
    result = run_ct("v2", client=ct_registry, data_root=releases)
    assert result.drift_status in {"ok", "drift"}
    report = mlflow.artifacts.load_dict(f"runs:/{result.run_id}/drift.json")
    assert report["champion_version"] == "1" and report["data_version"] == "v2"
    assert set(report["psi"]) == set(FEATURES)
    run = ct_registry.get_run(result.run_id)
    assert run.data.tags["drift_status"] == result.drift_status
    assert "drift_max_psi" in run.data.metrics


def test_ct_without_profile_still_decides(ct_registry, releases, monkeypatch):
    run_ct("v1", client=ct_registry, data_root=releases)

    # Champion trained before profiles existed: the artifact cannot be loaded.
    def missing(*args, **kwargs):
        raise MlflowException("no such artifact")

    monkeypatch.setattr("ml.ct_pipeline.mlflow.artifacts.load_dict", missing)
    result = run_ct("v2", client=ct_registry, data_root=releases)
    assert result.drift_status == "no_profile" and result.drift_max_psi is None
    assert result.decision in {"promoted", "rejected"}  # the decision is unaffected
    assert ct_registry.get_run(result.run_id).data.tags["drift_status"] == "no_profile"


def test_backfill_skips_existing_profile_unless_forced(ct_registry, releases):
    run_ct("v1", client=ct_registry, data_root=releases)
    run_id = ct_registry.get_model_version(MODEL, "1").run_id
    assert has_profile(ct_registry, run_id)  # written by the new training code

    skipped = backfill(MODEL, "1", client=ct_registry, data_root=releases)
    assert skipped == {"run_id": run_id, "written": False}
    out = backfill(MODEL, "1", client=ct_registry, data_root=releases, force=True)
    assert out["written"] is True
    assert ct_registry.get_run(run_id).data.tags["profile_backfilled"] == "true"
    profile = mlflow.artifacts.load_dict(f"runs:/{run_id}/{PROFILE_ARTIFACT}")
    assert profile["source"]["backfilled_for_version"] == "1"


def test_backfill_needs_the_release(ct_registry, releases, tmp_path):
    run_ct("v1", client=ct_registry, data_root=releases)
    write_release(tmp_path / "other", "v9", frame(4, 2, "x"), frame(4, 2, "y"))
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        backfill(MODEL, "1", client=ct_registry, data_root=tmp_path / "other", force=True)
