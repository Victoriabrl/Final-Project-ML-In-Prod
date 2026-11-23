"""Input validation on /predict: invalid bodies give 422, the 1000-character limit holds."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "body",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "\n\t "},
        {"text": "a" * 1001},
        {},
        {"message": "hello"},
        {"text": 123},
        {"text": 1.5},
        {"text": ["hello"]},
        {"text": None},
        {"text": {"nested": "hello"}},
    ],
    ids=[
        "empty",
        "spaces-only",
        "whitespace-only",
        "1001-chars",
        "missing-field",
        "wrong-field",
        "int",
        "float",
        "list",
        "null",
        "object",
    ],
)
def test_invalid_bodies_rejected(client, body):
    assert client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("raw", ['{"text": "hi"', "not json", ""])
def test_invalid_json_rejected(client, raw):
    r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_body_must_be_an_object(client):
    assert client.post("/predict", json=["hello"]).status_code == 422


def test_exactly_1000_chars_accepted(client):
    assert client.post("/predict", json={"text": "a" * 1000}).status_code == 200


def test_text_is_stripped_before_length_check(client):
    # 1000 useful characters surrounded by spaces: valid after cleaning.
    assert client.post("/predict", json={"text": "  " + "a" * 1000 + "  "}).status_code == 200


def test_single_character_accepted(client):
    assert client.post("/predict", json={"text": "a"}).status_code == 200
