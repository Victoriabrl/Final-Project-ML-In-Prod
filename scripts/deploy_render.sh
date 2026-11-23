#!/usr/bin/env sh
# Trigger a Render deploy of a specific image through the service's deploy hook.
#
# Usage: RENDER_DEPLOY_HOOK_URL=... scripts/deploy_render.sh ghcr.io/owner/repo:sha-<commit>
#
# Render docs (deploy hooks): GET or POST the hook URL; for an image-backed service,
# append the URL-encoded `imgURL` query parameter to deploy that tag instead of the
# one in the service settings. Everything but the tag must match the service's image.
# The hook URL is a secret: it is never printed.
set -eu

image="${1:?usage: deploy_render.sh IMAGE_URL}"
: "${RENDER_DEPLOY_HOOK_URL:?RENDER_DEPLOY_HOOK_URL is not set}"

encoded=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$image")

# The hook URL already carries ?key=..., so imgURL is appended with &.
status=$(curl -sS -o /tmp/render-deploy.json -w '%{http_code}' -X POST \
  "${RENDER_DEPLOY_HOOK_URL}&imgURL=${encoded}")

case "$status" in
  200) echo "Render deploy started for ${image}: $(cat /tmp/render-deploy.json)" ;;
  202) echo "Another deploy is in progress; Render queued ${image}" ;;
  *)
    echo "ERROR: Render deploy hook answered HTTP ${status} (400 = imgURL rejected," \
      "401/404 = invalid or regenerated hook URL, 409 = service suspended)" >&2
    exit 1
    ;;
esac
