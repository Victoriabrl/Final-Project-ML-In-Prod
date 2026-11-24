"""Grafana alerting provisioning (monitoring/alerting/): 3 rules, email contact point."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ALERTING = ROOT / "monitoring" / "alerting"


def load(name: str) -> dict:
    return yaml.safe_load((ALERTING / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules() -> dict:
    (group,) = load("rules.yml")["groups"]
    return {rule["title"]: rule for rule in group["rules"]}


def query(rule: dict) -> str:
    (prom,) = [d for d in rule["data"] if d["datasourceUid"] == "prometheus"]
    return " ".join(prom["model"]["expr"].split())


def threshold(rule: dict) -> tuple[str, float]:
    (cond,) = [d for d in rule["data"] if d["refId"] == rule["condition"]]
    evaluator = cond["model"]["conditions"][0]["evaluator"]
    return evaluator["type"], evaluator["params"][0]


def test_three_rules(rules):
    assert set(rules) == {"API down", "5xx error rate on /predict", "Input drift"}
    for rule in rules.values():
        assert rule["annotations"]["summary"], rule["title"]
        assert rule["uid"].startswith("sms-spam-api-")


def test_api_down_after_one_minute(rules):
    rule = rules["API down"]
    assert query(rule) == 'up{job="api"}'
    assert threshold(rule) == ("lt", 1)  # up == 0
    assert rule["for"] == "1m"
    assert rule["noDataState"] == "Alerting"  # no series = Prometheus cannot see the API


def test_error_rate_above_5_percent_on_predict(rules):
    rule = rules["5xx error rate on /predict"]
    expr = query(rule)
    assert 'path="/predict"' in expr and 'status=~"5.."' in expr and "[2m]" in expr
    assert threshold(rule) == ("gt", 5)
    assert rule["noDataState"] == "OK"  # no traffic or no 5xx is not an incident


def test_drift_at_threshold_for_two_minutes_says_investigate(rules):
    rule = rules["Input drift"]
    assert query(rule) == 'max(feature_drift{job="api"})'
    kind, value = threshold(rule)
    assert kind == "gt" and 0.199 <= value < 0.2  # i.e. PSI >= 0.20, as in app/drift.py
    assert rule["for"] == "2m"
    assert rule["noDataState"] == "OK"
    summary = rule["annotations"]["summary"].lower()
    assert "enquêter" in summary and "ne pas réentraîner" in summary


def test_email_contact_point_from_environment():
    (point,) = load("contact-points.yml")["contactPoints"]
    (receiver,) = point["receivers"]
    assert receiver["type"] == "email"
    assert receiver["settings"]["addresses"] == "${ALERT_EMAIL_TO}"  # set in .env only
    assert receiver["disableResolveMessage"] is False  # "resolved" emails too
    (policy,) = load("notification-policies.yml")["policies"]
    assert policy["receiver"] == point["name"]


def test_compose_provisions_the_alerting_folder():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    grafana = compose["services"]["grafana"]
    assert "./monitoring/alerting:/etc/grafana/provisioning/alerting:ro" in grafana["volumes"]
    assert str(grafana["environment"]["ALERT_EMAIL_TO"]).startswith("${")


def test_no_secret_in_alerting_files():
    text = " ".join(p.read_text(encoding="utf-8").lower() for p in ALERTING.glob("*.yml"))
    for word in ("password", "token", "smtp_pass", "@gmail", "mailtrap.io"):
        assert word not in text, word
