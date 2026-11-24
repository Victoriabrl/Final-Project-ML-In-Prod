"""Smoke test a running service: wait for /ready, check the model version, predict, and
check that /metrics reports the same model version (model_info) and the predictions.

Usage: python scripts/smoke_test.py --url URL [--expected-version N]
                                    [--expected-git-sha SHA] [--timeout 300]
Standard library only, so it runs anywhere (CI runner, laptop).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
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


def request_text(url: str, timeout: float = 30) -> tuple[int, str]:
    """GET a text endpoint (Prometheus /metrics)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def wait_ready(
    base: str,
    timeout: float,
    expected_git_sha: str | None = None,
    interval: float = 5,
    expected_version: str | None = None,
) -> dict:
    """Wait until /ready is 200 and, if given, serves the expected commit and model version.

    Render keeps the previous instance serving until the new one is healthy, so right
    after a deploy the URL still answers with the old instance. The commit check proves
    the new image is live; the model version check does the same when CT or rollback
    redeploy the same image to load another version of the model.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            if expected_git_sha is not None:
                status, health = request("GET", f"{base}/health", timeout=30)
                served_sha = health.get("git_sha") if isinstance(health, dict) else None
                if status != 200 or served_sha != expected_git_sha:
                    last = f"/health HTTP {status}, git_sha={served_sha!r}"
                    print(f"  old or no instance ({last}), retrying in {interval:.0f}s", flush=True)
                    time.sleep(interval)
                    continue
            status, body = request("GET", f"{base}/ready", timeout=30)
            served = body.get("model_version") if isinstance(body, dict) else None
            if status == 200 and (expected_version is None or served == str(expected_version)):
                return body
            if status == 200:
                last = f"/ready serves model version {served!r}, expected {expected_version!r}"
                print(f"  old instance ({last}), retrying in {interval:.0f}s", flush=True)
                time.sleep(interval)
                continue
            last = f"HTTP {status}: {str(body)[:200]}"
        # Not listening yet, cold start, or connection cut while an instance restarts.
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            last = repr(exc)
        print(f"  not ready yet ({last}), retrying in {interval:.0f}s", flush=True)
        time.sleep(interval)
    raise AssertionError(f"service not ready after {timeout:.0f}s (last: {last})")


def run(
    base: str,
    expected_version: str | None,
    timeout: float,
    expected_git_sha: str | None = None,
) -> list[str]:
    base = base.rstrip("/")
    lines = []

    def check(cond: bool, msg: str) -> None:
        lines.append(("PASS " if cond else "FAIL ") + msg)
        print(lines[-1], flush=True)
        if not cond:
            raise AssertionError(msg)

    print(f"Waiting for {base}/ready (timeout {timeout:.0f}s)", flush=True)
    ready = wait_ready(base, timeout, expected_git_sha, expected_version=expected_version)
    if expected_git_sha is not None:
        check(True, f"/health serves commit {expected_git_sha}")
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

    status, metrics = request_text(f"{base}/metrics")
    check(status == 200, f"/metrics HTTP {status}")
    info = f'model_info{{model_version="{served}"}} 1.0'
    check(info in metrics, f"/metrics {info}")
    counted = sum(
        float(value)
        for labels, value in re.findall(r"^predictions_total\{([^}]*)\} (\S+)$", metrics, re.M)
        if f'model_version="{served}"' in labels
    )
    check(counted >= 2, f"/metrics predictions_total for version {served} = {counted:.0f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-git-sha", help="wait until /health reports this commit")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)

    ok = True
    try:
        lines = run(args.url, args.expected_version, args.timeout, args.expected_git_sha)
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
