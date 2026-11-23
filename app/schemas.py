"""Pydantic request/response models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

MAX_TEXT_LENGTH = 1000

# strict=True: numbers, lists... are rejected instead of being coerced to str.
# The text is stripped first, then must contain 1 to 1000 characters.
SmsText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH),
]


class PredictRequest(BaseModel):
    text: SmsText


class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    spam_probability: float
    model_name: str
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    alias: str
    version: str | None
    run_id: str | None
    data_version: str | None
    loaded_at: str | None


class Status(BaseModel):
    status: str
    model_version: str | None = None
    detail: str | None = None
