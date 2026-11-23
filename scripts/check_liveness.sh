#!/usr/bin/env sh
# Bootstrap check when the registry has no approved champion yet: the container must
# start, report the expected commit on /health, and answer 503 on /ready (no model).
#
# Usage: scripts/check_liveness.sh BASE_URL EXPECTED_GIT_SHA [TIMEOUT_SECONDS]
set -eu

base="${1:?usage: check_liveness.sh BASE_URL EXPECTED_GIT_SHA [TIMEOUT]}"
sha="${2:?expected git sha}"
timeout="${3:-120}"

elapsed=0
until curl -fsS "${base}/health" 2>/dev/null | grep -q "\"git_sha\":\"${sha}\""; do
  if [ "$elapsed" -ge "$timeout" ]; then
    echo "FAIL /health never reported git_sha=${sha} within ${timeout}s" >&2
    exit 1
  fi
  sleep 3
  elapsed=$((elapsed + 3))
done
echo "PASS /health serves commit ${sha}"

code=$(curl -s -o /dev/null -w '%{http_code}' "${base}/ready")
if [ "$code" != "503" ]; then
  echo "FAIL /ready answered ${code}, expected 503 without an approved champion" >&2
  exit 1
fi
echo "PASS /ready is 503 (no approved champion to serve)"
