"""Input drift for the SMS model: simple text statistics + spam probability, compared with PSI.

One module shared by training (``training-profile.json``), the CT pipeline (``drift.json``)
and the API (``feature_drift`` gauge), so the three report values with the same meaning
and the same threshold. It lives in ``app/`` because the serving image only contains
``app/``; it needs nothing but numpy (already installed with scikit-learn).

Privacy: only these numbers are kept, never the text of a message.

PSI (Population Stability Index) per feature, with bins cut at the deciles of the
reference data: sum((actual - expected) * ln(actual / expected)) over the bins.
Common reading: < 0.10 stable, 0.10-0.20 moderate shift, >= 0.20 significant shift.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

PROFILE_ARTIFACT = "monitoring/training-profile.json"
PROFILE_SCHEMA = 1
DRIFT_THRESHOLD = 0.20
N_BINS = 10
EPS = 1e-4  # floor for empty bins, so the log stays finite

TEXT_FEATURES = ("length_chars", "digit_ratio", "upper_ratio", "has_url_or_phone")
SCORE_FEATURE = "spam_probability"
FEATURES = (*TEXT_FEATURES, SCORE_FEATURE)

# A link or a long run of digits (short codes, premium-rate or phone numbers).
_URL_OR_PHONE = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|co\.uk)\b|\d{5,}", re.I)


def text_features(text: str) -> dict[str, float]:
    """The 4 text statistics of one message (the text itself is not kept)."""
    n = len(text)
    letters = sum(c.isalpha() for c in text)
    return {
        "length_chars": float(n),
        "digit_ratio": sum(c.isdigit() for c in text) / n if n else 0.0,
        "upper_ratio": sum(c.isupper() for c in text) / letters if letters else 0.0,
        "has_url_or_phone": 1.0 if _URL_OR_PHONE.search(text) else 0.0,
    }


def text_feature_columns(texts: Iterable[str]) -> dict[str, np.ndarray]:
    rows = [text_features(t) for t in texts]
    return {name: np.array([r[name] for r in rows], dtype=float) for name in TEXT_FEATURES}


def _bin_edges(values: np.ndarray) -> list[float]:
    """Inner cut points: at the deciles, or between the distinct values when there are
    few of them (a binary feature that is 0 for 90% of messages would otherwise put
    every value in one bin and never drift).

    A value holding at least one decile of the reference (a point mass, e.g. 0 capitals
    or 0 digits) also gets a cut just above it, so "exactly that value" and "more than
    it" stay in separate bins; with a single distinct value, that cut is the only one.
    """
    if values.size == 0:
        return []
    distinct = np.unique(values)
    if distinct.size == 1:
        return [float(np.nextafter(distinct[0], np.inf))]
    if distinct.size <= N_BINS:
        return [float(x) for x in (distinct[:-1] + distinct[1:]) / 2]
    edges = set(np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1])).tolist())
    for value in list(edges):
        if np.mean(values == value) >= 1 / N_BINS:
            edges.add(float(np.nextafter(value, np.inf)))
    return sorted(float(x) for x in edges)


def _proportions(values: np.ndarray, edges: Sequence[float]) -> list[float]:
    counts = np.bincount(
        np.searchsorted(np.asarray(edges, dtype=float), values, side="right"),
        minlength=len(edges) + 1,
    )
    total = counts.sum()
    return [float(c / total) if total else 0.0 for c in counts]


def psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Population Stability Index between two binned distributions (same bins)."""
    value = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, EPS), max(a, EPS)
        value += (a - e) * math.log(a / e)
    return float(value)


def build_profile(
    texts: Iterable[str], scores: Iterable[float], source: Mapping[str, str] | None = None
) -> dict:
    """Reference distributions: text statistics of ``texts`` (the training split) and
    spam probabilities ``scores`` (the model on the held-out split, i.e. unseen data,
    like production traffic; scores on the training split would be overconfident)."""
    columns = text_feature_columns(texts)
    columns[SCORE_FEATURE] = np.asarray(list(scores), dtype=float)
    features = {}
    for name in FEATURES:
        values = columns[name]
        edges = _bin_edges(values)
        features[name] = {
            "edges": edges,
            "proportions": _proportions(values, edges),
            "mean": float(values.mean()) if values.size else None,
            "n": int(values.size),
        }
    return {
        "schema": PROFILE_SCHEMA,
        "threshold": DRIFT_THRESHOLD,
        "n_bins": N_BINS,
        "source": dict(source or {}),
        "features": features,
    }


def feature_psi(profile: Mapping, columns: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """PSI of each feature present in both the profile and ``columns``."""
    result = {}
    for name, ref in profile["features"].items():
        values = np.asarray(columns.get(name, []), dtype=float)
        if values.size == 0:
            continue
        actual = _proportions(values, ref["edges"])
        result[name] = psi(ref["proportions"], actual)
    return result


def drift_report(profile: Mapping, texts: Sequence[str], scores: Sequence[float]) -> dict:
    """Compare a batch of messages (and the model's scores on them) with a profile."""
    columns = text_feature_columns(texts)
    columns[SCORE_FEATURE] = np.asarray(list(scores), dtype=float)
    values = feature_psi(profile, columns)
    threshold = float(profile.get("threshold", DRIFT_THRESHOLD))
    return {
        "threshold": threshold,
        "n_rows": len(texts),
        "psi": values,
        "max_psi": max(values.values()) if values else 0.0,
        "drifted": sorted(name for name, v in values.items() if v >= threshold),
    }
