"""Model loading: alias resolution, readiness, failures and alias changes."""

from __future__ import annotations

from mlflow import MlflowClient

from tests.conftest import UNREACHABLE_URI, wait_for


def test_loads_champion_not_newest_version(client, registry):
    mlflow_client = MlflowClient(registry["tracking_uri"])
    versions = {str(mv.version) for mv in mlflow_client.search_model_versions("name='sms-spam'")}
    assert registry["newer_version"] in versions  # a newer candidate exists...

    info = client.get("/model-info").json()
    assert info["version"] == registry["champion_version"] == "1"  # ...but v1 is served


def test_model_info_fields(client, registry):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert set(info) == {"model_name", "alias", "version", "run_id", "data_version", "loaded_at"}
    assert info["model_name"] == "sms-spam"
    assert info["alias"] == "champion"
    assert info["data_version"] == "v1"
    assert info["run_id"]
    assert info["loaded_at"]


def test_health_and_ready_when_loaded(client):
    assert client.get("/health").status_code == 200
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == "1"


def _assert_not_ready(client):
    loader = client.app.state.loader
    assert wait_for(lambda: loader.last_error is not None), "no loading attempt recorded"
    assert client.get("/health").status_code == 200  # liveness unaffected
    assert client.get("/ready").status_code == 503
    assert client.get("/model-info").status_code == 503
    r = client.post("/predict", json={"text": "hello"})
    assert r.status_code == 503
    assert "label" not in r.json()  # never a prediction without a model


def test_missing_alias_not_ready(registry, make_client):
    client = make_client(registry["tracking_uri"], alias="does-not-exist", wait_ready=False)
    _assert_not_ready(client)


def test_unreachable_registry_not_ready(make_client):
    client = make_client(UNREACHABLE_URI, wait_ready=False)
    _assert_not_ready(client)


def test_unapproved_champion_refused(fresh_registry, make_client):
    mlflow_client = MlflowClient(fresh_registry["tracking_uri"])
    mlflow_client.set_model_version_tag("sms-spam", "1", "status", "candidate")
    client = make_client(fresh_registry["tracking_uri"], wait_ready=False)
    _assert_not_ready(client)
    assert "approved" in client.app.state.loader.last_error


def test_alias_move_then_reload_serves_new_version(fresh_registry, make_client):
    uri = fresh_registry["tracking_uri"]
    client = make_client(uri)
    assert client.get("/model-info").json()["version"] == "1"

    mlflow_client = MlflowClient(uri)
    mlflow_client.set_model_version_tag("sms-spam", "2", "status", "approved")
    mlflow_client.set_registered_model_alias("sms-spam", "champion", "2")

    # The running process keeps its resolved version until it reloads...
    assert client.get("/model-info").json()["version"] == "1"

    # ...in-process reload resolves the alias again.
    client.app.state.loader.load()
    assert client.get("/model-info").json()["version"] == "2"
    assert client.post("/predict", json={"text": "hi"}).json()["model_version"] == "2"

    # A restarted process (what a redeploy does) also serves version 2.
    restarted = make_client(uri)
    assert restarted.get("/model-info").json()["version"] == "2"


def test_failed_reload_keeps_serving_previous_model(fresh_registry, make_client):
    uri = fresh_registry["tracking_uri"]
    client = make_client(uri)
    # Alias moved to a version that is not approved: the reload must fail...
    MlflowClient(uri).set_registered_model_alias("sms-spam", "champion", "2")
    assert client.app.state.loader.try_load() is False
    # ...and the approved version 1 keeps being served.
    assert client.get("/ready").status_code == 200
    assert client.get("/model-info").json()["version"] == "1"
