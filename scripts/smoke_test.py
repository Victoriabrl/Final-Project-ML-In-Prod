"""Smoke test a running service: wait for /ready, check the model version, predict.

Usage: python scripts/smoke_test.py --url URL [--expected-version N] [--timeout 300]
Standard library only, so it runs anywhere (CI runner, laptop).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SPAM_TEXT = "WINNER!! You have won a free prize worth 1000 cash. Call 09061701461 now to claim"
HAM_TEXT = "Hey, are we still meeting for lunch tomorrow? Let me know"


def request(method: str, url: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        try:
            return exc.code, json.loads(payload)
        except ValueError:
            return exc.code, payload.decode(errors="replace")


def wait_ready(base: str, timeout: float, interval: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            status, body = request("GET", f"{base}/ready", timeout=30)
            if status == 200:
                return body
            last = f"HTTP {status}: {str(body)[:200]}"
        except (urllib.error.URLError, OSError) as exc:  # not listening yet / cold start
            last = repr(exc)
        print(f"  not ready yet ({last}), retrying in {interval:.0f}s", flush=True)
        time.sleep(interval)
    raise AssertionError(f"service not ready after {timeout:.0f}s (last: {last})")


def run(base: str, expected_version: str | None, timeout: float) -> list[str]:
    base = base.rstrip("/")
    lines = []

    def check(cond: bool, msg: str) -> None:
        lines.append(("PASS " if cond else "FAIL ") + msg)
        print(lines[-1], flush=True)
        if not cond:
            raise AssertionError(msg)

    print(f"Waiting for {base}/ready (timeout {timeout:.0f}s)", flush=True)
    ready = wait_ready(base, timeout)
    check(True, f"/ready -> {ready}")

    status, info = request("GET", f"{base}/model-info")
    check(status == 200, f"/model-info HTTP {status}: {info}")
    served = str(info["version"])
    if expected_version is not None:
        check(served == str(expected_version), f"served version {served} == {expected_version}")

    for text, expected in ((SPAM_TEXT, "spam"), (HAM_TEXT, "ham")):
        status, pred = request("POST", f"{base}/predict", {"text": text})
        check(status == 200, f"/predict HTTP {status}")
        check(pred["label"] == expected, f"{expected} example -> {pred['label']}")
        check(0.0 <= pred["spam_probability"] <= 1.0, f"probability {pred['spam_probability']}")
        check(pred["model_version"] == served, f"response model_version {pred['model_version']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)

    ok = True
    try:
        lines = run(args.url, args.expected_version, args.timeout)
    except (AssertionError, KeyError, urllib.error.URLError, OSError) as exc:
        ok, lines = False, [f"FAIL {exc}"]
        print(lines[0], flush=True)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Smoke test {args.url}: {'✅ passed' if ok else '❌ failed'}\n\n")
            fh.write("\n".join(f"- {line}" for line in lines) + "\n\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
