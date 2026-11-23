"""Print the concrete version behind the champion alias (expected version for smoke tests).

Usage: python -m scripts.resolve_champion [--model-name sms-spam] [--alias champion]
Reads MLFLOW_TRACKING_URI / credentials from the environment. Fails (exit 1) when the
alias does not exist or points to a version that is not approved: the app would refuse
to serve it, so deploying would be pointless. In GitHub Actions the version is also
written to $GITHUB_OUTPUT as ``version=N``.
"""

from __future__ import annotations

import argparse
import os
import sys

from mlflow import MlflowClient

from ml.registry_utils import (
    CHAMPION_ALIAS,
    STATUS_APPROVED,
    STATUS_TAG,
    get_client,
    get_version_by_alias,
)


def resolve(model_name: str, alias: str = CHAMPION_ALIAS, client=None) -> str:
    client = client or get_client()
    mv = get_version_by_alias(client, model_name, alias)
    if mv is None:
        raise LookupError(f"no '{alias}' alias on model '{model_name}'")
    status = mv.tags.get(STATUS_TAG)
    if status != STATUS_APPROVED:
        raise LookupError(f"{model_name}@{alias} -> version {mv.version} has status {status!r}")
    return str(mv.version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "sms-spam"))
    parser.add_argument("--alias", default=os.environ.get("MODEL_ALIAS", CHAMPION_ALIAS))
    parser.add_argument("--tracking-uri", help="default: MLFLOW_TRACKING_URI")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 with found=false when there is no approved champion yet (bootstrap)",
    )
    args = parser.parse_args(argv)
    client = MlflowClient(args.tracking_uri) if args.tracking_uri else None
    output = os.environ.get("GITHUB_OUTPUT")
    try:
        version = resolve(args.model_name, args.alias, client=client)
    except LookupError as exc:
        if not args.allow_missing:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"WARNING: {exc} (no approved champion yet)", file=sys.stderr)
        if output:
            with open(output, "a", encoding="utf-8") as fh:
                fh.write("found=false\n")
        return 0
    print(version)
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"found=true\nversion={version}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
