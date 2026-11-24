"""Replay SMS from a CSV against the API's POST /predict, to generate monitored traffic.

Usage:
  python scripts/replay.py --file .exercise-data/normal.csv [--url http://127.0.0.1:8000]
                           [--rate 10] [--limit 500] [--shuffle] [--seed 42]
                           [--broken] [--broken-ratio 0.2]

Each request carries its own X-Request-ID, so one prediction can be followed in the logs.
``--broken`` mixes invalid requests in (empty text, missing field, text over 1000
characters, malformed JSON): the API must answer them with 4xx, never 5xx.
At the end: number of responses per HTTP status, predicted labels, served model versions
and latency. Standard library only, so it runs anywhere (laptop, CI runner).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

MAX_TEXT_LENGTH = 1000  # same limit as app/schemas.py

# Invalid payloads (raw request bodies) sent in --broken mode.
BROKEN_BODIES: dict[str, bytes] = {
    "empty_text": json.dumps({"text": ""}).encode(),
    "missing_field": json.dumps({"message": "hello"}).encode(),
    "too_long": json.dumps({"text": "a" * (MAX_TEXT_LENGTH + 1)}).encode(),
    "malformed_json": b'{"text": "unterminated',
}

# send(body, request_id) -> (HTTP status or 0 on network error, parsed JSON or None)
Sender = Callable[[bytes, str], tuple[int, dict | None]]


def http_sender(url: str, timeout: float = 30) -> Sender:
    endpoint = url.rstrip("/") + "/predict"

    def send(body: bytes, request_id: str) -> tuple[int, dict | None]:
        req = urllib.request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Request-ID", request_id)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.status, json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"null")
            except ValueError:
                return exc.code, None
        except (urllib.error.URLError, OSError):
            return 0, None

    return send


def read_texts(path: Path) -> list[str]:
    """The ``text`` column of a CSV (other columns, e.g. ``label``, are ignored)."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{path}: a 'text' column is required")
        return [row["text"] for row in reader]


def build_requests(
    texts: list[str],
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    broken_ratio: float = 0.0,
) -> list[tuple[str, bytes]]:
    """(kind, body) pairs; kind is 'valid' or the name of a broken payload."""
    rng = random.Random(seed)
    texts = list(texts)
    if shuffle:
        rng.shuffle(texts)
    if limit is not None:
        texts = texts[:limit]
    broken_kinds = sorted(BROKEN_BODIES)
    requests = []
    for i, text in enumerate(texts):
        if broken_ratio and rng.random() < broken_ratio:
            kind = broken_kinds[i % len(broken_kinds)]
            requests.append((kind, BROKEN_BODIES[kind]))
        else:
            requests.append(("valid", json.dumps({"text": text}).encode()))
    return requests


@dataclass
class ReplayReport:
    statuses: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    model_versions: Counter = field(default_factory=Counter)
    kinds: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)

    def percentile(self, q: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    def summary(self) -> dict:
        return {
            "requests": sum(self.statuses.values()),
            "by_status": {str(k): v for k, v in sorted(self.statuses.items())},
            "by_kind": dict(sorted(self.kinds.items())),
            "labels": dict(sorted(self.labels.items())),
            "model_versions": dict(sorted(self.model_versions.items())),
            "latency_ms_p50": round(self.percentile(0.50), 1),
            "latency_ms_p95": round(self.percentile(0.95), 1),
        }


def replay(
    requests: Iterable[tuple[str, bytes]],
    send: Sender,
    rate: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    progress_every: int = 100,
) -> ReplayReport:
    """Send the requests in order, at most ``rate`` per second (0 = as fast as possible)."""
    report = ReplayReport()
    interval = 1.0 / rate if rate > 0 else 0.0
    for i, (kind, body) in enumerate(requests, start=1):
        started = time.monotonic()
        request_id = str(uuid.uuid4())
        status, payload = send(body, request_id)
        elapsed = time.monotonic() - started
        report.request_ids.append(request_id)
        report.kinds[kind] += 1
        report.statuses[status] += 1
        report.latencies_ms.append(elapsed * 1000)
        if status == 200 and isinstance(payload, dict):
            report.labels[payload.get("label", "?")] += 1
            report.model_versions[str(payload.get("model_version", "?"))] += 1
        if progress_every and i % progress_every == 0:
            print(f"  {i} requests sent, statuses so far: {dict(report.statuses)}", flush=True)
        if interval > elapsed:
            sleep(interval - elapsed)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="CSV with a 'text' column")
    # 127.0.0.1, not localhost: on Windows localhost tries IPv6 first (~2 s per request).
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=10.0, help="requests/s, 0 = no limit")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--broken", action="store_true", help="mix invalid requests in")
    parser.add_argument("--broken-ratio", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)

    texts = read_texts(args.file)
    requests = build_requests(
        texts,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        broken_ratio=args.broken_ratio if args.broken else 0.0,
    )
    print(f"Replaying {len(requests)} requests from {args.file} to {args.url}", flush=True)
    report = replay(requests, http_sender(args.url, args.timeout), rate=args.rate)
    summary = report.summary()
    print(json.dumps(summary, indent=2))
    # Exit code 1 if the API failed (5xx or unreachable): 4xx are expected in --broken mode.
    failed = sum(n for status, n in report.statuses.items() if status == 0 or status >= 500)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
