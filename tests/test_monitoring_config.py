"""Monitoring stack files: docker-compose.yml, Prometheus, Grafana provisioning, dashboard.

Static checks (no Docker needed); CI also runs `promtool check config` and
`docker compose config` on the same files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "monitoring"
METRICS = {
    "http_requests_total",
    "http_request_duration_seconds",
    "predictions_total",
    "model_info",
    "feature_drift",
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose() -> dict:
    return load_yaml(ROOT / "docker-compose.yml")


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads((MON / "dashboard.json").read_text(encoding="utf-8"))


def test_compose_has_the_three_services_with_pinned_images(compose):
    services = compose["services"]
    assert set(services) == {"api", "prometheus", "grafana"}
    assert services["api"]["build"]["context"] == "."  # same Dockerfile as production
    for name in ("prometheus", "grafana"):
        image = services[name]["image"]
        assert re.fullmatch(r"[\w./-]+:v?\d+\.\d+\.\d+", image), f"{name} image not pinned: {image}"


def test_compose_takes_every_credential_from_the_environment(compose):
    env = {
        **compose["services"]["api"]["environment"],
        **compose["services"]["grafana"]["environment"],
    }
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "GF_SMTP_HOST",
        "GF_SMTP_USER",
        "GF_SMTP_PASSWORD",
        "GF_SECURITY_ADMIN_PASSWORD",
    ):
        assert str(env[key]).startswith("${"), f"{key} must come from .env, not be written here"


def test_compose_mounts_existing_provisioning_files(compose):
    for service in ("prometheus", "grafana"):
        for volume in compose["services"][service]["volumes"]:
            volume = volume.replace("${PROMETHEUS_CONFIG:-prometheus.yml}", "prometheus.yml")
            source = volume.split(":", 1)[0]
            if source.startswith("./"):
                assert (ROOT / source).exists(), source


@pytest.mark.parametrize("name", ["prometheus.yml", "prometheus.prod.yml"])
def test_prometheus_scrapes_the_api_job(name):
    config = load_yaml(MON / name)
    (job,) = config["scrape_configs"]
    assert job["job_name"] == "api"  # the dashboard and the alerts query up{job="api"}
    assert job["metrics_path"] == "/metrics"
    if name == "prometheus.yml":
        assert config["global"]["scrape_interval"] == "5s"
        assert job["static_configs"][0]["targets"] == ["api:8000"]
    else:
        assert job["scheme"] == "https"


def test_datasource_matches_the_dashboard(dashboard):
    (source,) = load_yaml(MON / "grafana-datasource.yml")["datasources"]
    assert source["url"] == "http://prometheus:9090" and source["type"] == "prometheus"
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == source["uid"]


def test_dashboard_has_the_six_panels(dashboard):
    assert dashboard["uid"] == "sms-spam-api"
    panels = dashboard["panels"]
    assert len(panels) == 6
    exprs = " ".join(t["expr"] for p in panels for t in p["targets"])
    for metric in METRICS:
        assert metric in exprs, metric
    titles = " ".join(p["title"].lower() for p in panels)
    for word in ("requests", "5xx", "p95", "class", "drift", "version"):
        assert word in titles, word


def test_request_panels_filter_on_predict_and_use_rate(dashboard):
    for panel in dashboard["panels"]:
        for t in panel["targets"]:
            if "http_request" in t["expr"]:
                assert 'path="/predict"' in t["expr"], panel["title"]
            if "_total" in t["expr"]:
                # Counters restart at 0 with every new instance: never the raw value.
                assert "rate(" in t["expr"], panel["title"]


def test_drift_panel_draws_the_threshold_line(dashboard):
    (drift,) = [p for p in dashboard["panels"] if "feature_drift" in p["targets"][0]["expr"]]
    defaults = drift["fieldConfig"]["defaults"]
    assert {"color": "red", "value": 0.2} in defaults["thresholds"]["steps"]
    assert defaults["custom"]["thresholdsStyle"]["mode"] == "line"


def test_no_secret_in_the_exported_dashboard():
    text = (MON / "dashboard.json").read_text(encoding="utf-8").lower()
    for word in ("password", "token", "apikey", "secret", "dagshub", "@"):
        assert word not in text, word
