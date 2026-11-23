"""Prediction endpoint: labels, response format, probability range, served version."""

from __future__ import annotations

import pytest

SPAM = "WINNER! You have won a free cash prize, call now to claim"
HAM = "Are we still meeting for lunch tomorrow?"


def test_obvious_spam_is_spam(client):
    r = client.post("/predict", json={"text": SPAM})
    assert r.status_code == 200
    assert r.json()["label"] == "spam"


def test_obvious_ham_is_ham(client):
    r = client.post("/predict", json={"text": HAM})
    assert r.status_code == 200
    assert r.json()["label"] == "ham"


def test_response_has_exact_format(client, registry):
    body = client.post("/predict", json={"text": SPAM}).json()
    assert set(body) == {"label", "spam_probability", "model_name", "model_version"}
    assert body["label"] in {"spam", "ham"}
    assert isinstance(body["spam_probability"], float)
    assert body["model_name"] == registry["model_name"]
    assert isinstance(body["model_version"], str)


@pytest.mark.parametrize("text", [SPAM, HAM, "x"])
def test_probability_in_unit_interval(client, text):
    p = client.post("/predict", json={"text": text}).json()["spam_probability"]
    assert 0.0 <= p <= 1.0


def test_spam_probability_higher_for_spam(client):
    p_spam = client.post("/predict", json={"text": SPAM}).json()["spam_probability"]
    p_ham = client.post("/predict", json={"text": HAM}).json()["spam_probability"]
    assert p_spam > 0.5 > p_ham


def test_returned_version_is_champion(client, registry):
    body = client.post("/predict", json={"text": HAM}).json()
    assert body["model_version"] == registry["champion_version"]


@pytest.mark.parametrize(
    "text",
    [
        "Café à 15h ? Ça marche, à tout à l'heure",
        "Happy birthday!! 🎉🎂 see you tonight 😀",
        "Call 0800 123 456 before 23:59 on 31/12",
        "ÄÖÜ ß — “quotes” and ‘apostrophes’",
        "😀",
    ],
)
def test_accents_emojis_and_digits_accepted(client, text):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200
    assert r.json()["label"] in {"spam", "ham"}


def test_intentionally_broken():
    assert 1 == 2  # demo: a failing PR must not be mergeable
