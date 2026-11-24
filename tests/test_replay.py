"""Replay tooling: scripts/replay.py (traffic generator) and ml/make_replay_sets.py."""

from __future__ import annotations

import json

import pytest

from app.drift import DRIFT_THRESHOLD, build_profile, drift_report
from ml.data_utils import read_csv
from ml.make_replay_sets import make_replay_sets
from scripts.replay import BROKEN_BODIES, build_requests, read_texts, replay
from scripts.replay import main as replay_main
from tests.conftest import UNREACHABLE_URI, frame, write_release

TEXTS = [f"see you at lunch day {i}" for i in range(40)] + ["WIN a FREE prize, call 0800"] * 10


def app_sender(client):
    """Send through the FastAPI TestClient (real app, test registry, no network)."""

    def send(body: bytes, request_id: str):
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        resp = client.post("/predict", content=body, headers=headers)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, None

    return send


def test_read_texts_needs_a_text_column(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text('label,text\nham,"hi, there"\nspam,WIN\n', encoding="utf-8")
    assert read_texts(good) == ["hi, there", "WIN"]
    bad = tmp_path / "bad.csv"
    bad.write_text("label,message\nham,hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'text' column"):
        read_texts(bad)


def test_build_requests_limit_shuffle_and_broken():
    plain = build_requests(TEXTS, limit=5)
    assert [json.loads(body)["text"] for _, body in plain] == TEXTS[:5]
    # Shuffling is reproducible with the same seed, different with another one.
    first = build_requests(TEXTS, shuffle=True, seed=1)
    assert first == build_requests(TEXTS, shuffle=True, seed=1)
    assert first != build_requests(TEXTS, shuffle=True, seed=2)
    assert first != build_requests(TEXTS)  # really shuffled
    assert {kind for kind, _ in build_requests(TEXTS)} == {"valid"}
    broken = build_requests(TEXTS, broken_ratio=0.5)
    kinds = {kind for kind, _ in broken}
    assert kinds - {"valid"} and kinds - {"valid"} <= set(BROKEN_BODIES)


def test_replay_against_the_app(client, registry):
    requests = build_requests(TEXTS, broken_ratio=0.3, seed=3)
    report = replay(requests, app_sender(client), progress_every=0)
    summary = report.summary()

    assert summary["requests"] == len(TEXTS)
    # Valid SMS -> 200, invalid payloads -> 422, never a server error.
    assert set(report.statuses) == {200, 422}
    assert report.statuses[200] == report.kinds["valid"]
    assert sum(report.labels.values()) == report.statuses[200]
    assert set(report.model_versions) == {registry["champion_version"]}
    # One X-Request-ID per request.
    assert len(set(report.request_ids)) == len(TEXTS)


def test_rate_limits_the_sending_speed():
    sleeps: list[float] = []
    requests = build_requests(TEXTS, limit=5)
    replay(requests, lambda body, rid: (200, {}), rate=10, sleep=sleeps.append, progress_every=0)
    assert len(sleeps) == 5 and all(0.09 < s <= 0.1 for s in sleeps)


def test_cli_fails_when_the_api_is_down(tmp_path, capsys):
    path = tmp_path / "texts.csv"
    path.write_text("label,text\nham,hello\n", encoding="utf-8")
    code = replay_main(
        ["--file", str(path), "--url", UNREACHABLE_URI, "--rate", "0", "--timeout", "2"]
    )
    assert code == 1
    assert '"0": 1' in capsys.readouterr().out  # status 0 = unreachable


@pytest.fixture
def release_root(tmp_path):
    root = tmp_path / "releases"
    write_release(root, "v2", frame(60, 20, "train"), frame(30, 10, "holdout"))
    return root


def test_replay_sets_are_reproducible(release_root, tmp_path):
    a = make_replay_sets("v2", release_root, tmp_path / "a", campaign_size=50)
    b = make_replay_sets("v2", release_root, tmp_path / "b", campaign_size=50)
    assert a["files"] == b["files"]  # same rows, same sha256

    normal = read_csv(tmp_path / "a" / "normal.csv")
    campaign = read_csv(tmp_path / "a" / "campaign.csv")
    assert len(normal) == 40 and set(normal["label"]) == {"ham", "spam"}
    assert len(campaign) == 50 and set(campaign["label"]) == {"spam"}
    assert (campaign["text"] == campaign["text"].str.upper()).mean() == pytest.approx(0.5)


def test_campaign_drifts_and_normal_does_not(release_root, tmp_path):
    make_replay_sets("v2", release_root, tmp_path / "sets", campaign_size=200)
    train = read_csv(release_root / "v2" / "train.csv")
    profile = build_profile(train["text"], [0.1] * len(train))

    normal = read_csv(tmp_path / "sets" / "normal.csv")["text"].tolist()
    campaign = read_csv(tmp_path / "sets" / "campaign.csv")["text"].tolist()
    # Text statistics only (no model here): compare with scores equal to the profile's.
    assert drift_report(profile, campaign, [0.1] * len(campaign))["max_psi"] >= DRIFT_THRESHOLD
    calm = drift_report(profile, normal, [0.1] * len(normal))
    assert "upper_ratio" not in calm["drifted"]


def test_replay_sets_need_the_release(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        make_replay_sets("v2", tmp_path / "nothing", tmp_path / "out")
