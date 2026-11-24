"""Live drift (app/drift_monitor.py): window of 500, recompute every 100, feature_drift."""

from __future__ import annotations

import random

import pytest
from mlflow import MlflowClient
from prometheus_client.parser import text_string_to_metric_families

from app.drift import DRIFT_THRESHOLD, FEATURES, PROFILE_ARTIFACT, build_profile
from app.drift_monitor import DriftMonitor

WORDS = [
    "see",
    "you",
    "at",
    "lunch",
    "tomorrow",
    "ok",
    "thanks",
    "call",
    "me",
    "later",
    "home",
    "now",
    "love",
    "the",
    "meeting",
    "is",
]


def normal_messages(n: int, seed: int) -> list[tuple[str, float]]:
    """Everyday traffic: short lowercase messages, a few with a number (score ~0.1)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 20)))
        if rng.random() < 0.1:
            text += f" call {rng.randint(10000, 99999)}"
        out.append((text, rng.uniform(0.0, 0.2)))
    return out


def campaign_messages(n: int, seed: int) -> list[tuple[str, float]]:
    """A spam campaign: long, capitals, numbers and links, high scores."""
    rng = random.Random(seed)
    return [
        (
            f"URGENT!! YOU WON {rng.randint(100, 9999)} GBP CASH. "
            f"CALL 0906{rng.randint(1000000, 9999999)} "
            f"OR TXT WIN TO 8{rng.randint(1000, 9999)} NOW WWW.CLAIM-PRIZE.COM",
            rng.uniform(0.9, 1.0),
        )
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def profile() -> dict:
    reference = normal_messages(3000, seed=0)
    return build_profile([t for t, _ in reference], [s for _, s in reference])


def feed(monitor: DriftMonitor, messages) -> None:
    for text, score in messages:
        monitor.observe(text, score)


def test_no_value_before_the_window_is_full(profile):
    monitor = DriftMonitor(profile, "5", window_size=500, recompute_every=100)
    feed(monitor, normal_messages(499, seed=1))
    assert monitor.latest == {} and monitor.computations == 0
    monitor.observe("see you at lunch", 0.05)  # 500th prediction
    assert monitor.computations == 1
    assert set(monitor.latest) == set(FEATURES)
    assert max(monitor.latest.values()) < DRIFT_THRESHOLD  # same distribution


def test_recomputed_only_every_100_predictions(profile):
    monitor = DriftMonitor(profile, "5", window_size=500, recompute_every=100)
    feed(monitor, normal_messages(599, seed=2))
    assert monitor.computations == 1  # at 500
    monitor.observe("ok thanks", 0.02)
    assert monitor.computations == 2  # at 600


def test_window_is_capped_and_keeps_no_text(profile):
    monitor = DriftMonitor(profile, "5", window_size=500, recompute_every=100)
    feed(monitor, normal_messages(1234, seed=3))
    assert len(monitor.window) == 500
    assert monitor.computations == 8  # 500, 600, ..., 1200
    assert all(set(row) == set(FEATURES) for row in monitor.window)
    assert all(isinstance(v, float) for row in monitor.window for v in row.values())


def test_campaign_crosses_the_threshold_then_recovers(profile):
    monitor = DriftMonitor(profile, "5", window_size=500, recompute_every=100)
    feed(monitor, normal_messages(500, seed=4))
    assert max(monitor.latest.values()) < DRIFT_THRESHOLD
    feed(monitor, campaign_messages(500, seed=5))
    drifted = {k for k, v in monitor.latest.items() if v >= DRIFT_THRESHOLD}
    assert {"upper_ratio", "digit_ratio", "spam_probability"} <= drifted
    feed(monitor, normal_messages(500, seed=6))
    assert max(monitor.latest.values()) < DRIFT_THRESHOLD


def test_invalid_sizes_are_refused(profile):
    with pytest.raises(ValueError):
        DriftMonitor(profile, "5", window_size=0)
    with pytest.raises(ValueError):
        DriftMonitor(profile, "5", recompute_every=0)


# --- in the app: profile downloaded with the champion, gauge on /metrics ---------------


def feature_drift(client) -> dict[str, float]:
    text = client.get("/metrics").text
    return {
        s.labels["feature"]: s.value
        for family in text_string_to_metric_families(text)
        if family.name == "feature_drift"
        for s in family.samples
    }


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setenv("DRIFT_WINDOW_SIZE", "50")
    monkeypatch.setenv("DRIFT_RECOMPUTE_EVERY", "10")


def test_app_exposes_feature_drift_from_the_champion_profile(
    fresh_registry, make_client, small_window, profile
):
    uri = fresh_registry["tracking_uri"]
    mlflow_client = MlflowClient(uri)
    run_id = mlflow_client.get_model_version("sms-spam", fresh_registry["champion_version"]).run_id
    mlflow_client.log_dict(run_id, profile, PROFILE_ARTIFACT)

    client = make_client(uri)
    for text, _ in normal_messages(49, seed=7):
        assert client.post("/predict", json={"text": text}).status_code == 200
    assert feature_drift(client) == {}  # window not full yet
    client.post("/predict", json={"text": "see you at lunch"})
    values = feature_drift(client)
    assert set(values) == set(FEATURES)

    for text, _ in campaign_messages(50, seed=8):
        client.post("/predict", json={"text": text})
    after = feature_drift(client)
    assert max(after.values()) >= DRIFT_THRESHOLD
    assert after["upper_ratio"] >= DRIFT_THRESHOLD


def test_app_without_profile_serves_without_drift(registry, make_client, small_window):
    # The shared test registry has no training profile: the service is still ready.
    client = make_client(registry["tracking_uri"])
    for text, _ in normal_messages(60, seed=9):
        assert client.post("/predict", json={"text": text}).status_code == 200
    assert feature_drift(client) == {}
    assert client.get("/ready").status_code == 200


def test_point_mass_in_the_profile_can_still_drift():
    # Reference: never any capital letter (upper_ratio is always 0).
    reference = ["see you at lunch", "ok thanks", "call me later"] * 100
    prof = build_profile(reference, [0.1] * len(reference))
    assert prof["features"]["upper_ratio"]["edges"] != []
    shouting = ["SEE YOU AT LUNCH", "OK THANKS"] * 100
    from app.drift import drift_report

    assert "upper_ratio" in drift_report(prof, shouting, [0.1] * 200)["drifted"]
    assert drift_report(prof, reference, [0.1] * 300)["max_psi"] < DRIFT_THRESHOLD
