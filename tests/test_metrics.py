"""Prometheus /metrics: the 5 metrics, counters that move, model_info, low cardinality."""

from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from tests.conftest import UNREACHABLE_URI

METRIC_NAMES = {
    "http_requests_total",
    "http_request_duration_seconds",
    "predictions_total",
    "model_info",
    "feature_drift",
}


def scrape(client) -> dict:
    """{sample name: [(labels, value), ...]} from GET /metrics."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    samples: dict[str, list] = {}
    for family in text_string_to_metric_families(resp.text):
        for sample in family.samples:
            samples.setdefault(sample.name, []).append((sample.labels, sample.value))
    return samples


def value(samples: dict, name: str, **labels) -> float:
    return sum(
        v for lab, v in samples.get(name, []) if all(lab.get(k) == x for k, x in labels.items())
    )


def test_metrics_lists_the_five_metrics(client):
    text = client.get("/metrics").text
    for name in METRIC_NAMES:
        assert f"# TYPE {name} " in text, name


def test_counters_increase_after_a_prediction(client, registry):
    version = registry["champion_version"]
    before = scrape(client)
    resp = client.post("/predict", json={"text": "WIN a free prize, call now"})
    assert resp.status_code == 200
    label = resp.json()["label"]
    after = scrape(client)

    assert value(after, "predictions_total", prediction=label, model_version=version) == (
        value(before, "predictions_total", prediction=label, model_version=version) + 1
    )
    req = {"method": "POST", "path": "/predict", "status": "200"}
    assert (
        value(after, "http_requests_total", **req)
        == value(before, "http_requests_total", **req) + 1
    )
    assert value(after, "http_request_duration_seconds_count", path="/predict") >= 1


def test_model_info_matches_the_loaded_version(client):
    served = client.get("/model-info").json()["version"]
    info = scrape(client)["model_info"]
    assert info == [({"model_version": served}, 1.0)]


def test_metrics_answer_before_the_model_but_without_model_info(make_client):
    client = make_client(UNREACHABLE_URI, wait_ready=False)
    assert client.get("/ready").status_code == 503
    samples = scrape(client)  # /metrics still answers 200
    assert "model_info" not in samples
    assert client.post("/predict", json={"text": "hello"}).status_code == 503
    assert value(scrape(client), "http_requests_total", path="/predict", status="503") == 1


def test_errors_and_paths_use_route_templates(client):
    client.post("/predict", json={"text": ""})  # 422
    client.get("/model-info?debug=1", headers={"X-Request-ID": "never-a-label"})
    client.get("/no/such/page/123")
    samples = scrape(client)
    assert value(samples, "http_requests_total", path="/predict", status="422") >= 1
    paths = {lab["path"] for lab, _ in samples["http_requests_total"]}
    assert "/model-info" in paths and "<unmatched>" in paths
    assert not any("?" in p or "123" in p for p in paths)


def test_labels_stay_low_cardinality(client):
    for i in range(5):
        client.post(
            "/predict", json={"text": f"message number {i}"}, headers={"X-Request-ID": f"id-{i}"}
        )
    text = client.get("/metrics").text
    assert "id-" not in text and "message number" not in text
    allowed = {
        "http_requests_total": {"method", "path", "status"},
        "http_request_duration_seconds_bucket": {"path", "le"},
        "predictions_total": {"prediction", "model_version"},
        "model_info": {"model_version"},
    }
    samples = scrape(client)
    for name, keys in allowed.items():
        for labels, _ in samples.get(name, []):
            assert set(labels) == keys, (name, labels)
    predicted = {lab["prediction"] for lab, _ in samples["predictions_total"]}
    assert predicted <= {"spam", "ham"}
