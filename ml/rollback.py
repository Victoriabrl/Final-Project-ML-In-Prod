"""Move the champion alias back to an approved version (rollback.yml, or by hand).

Usage:
  python -m ml.rollback --model-name sms-spam [--version N] [--reason "..."]

Without ``--version`` the target is the ``previous_champion`` tag of the current
champion. A version without ``status=approved`` is refused: the app would not serve it.
The operation is logged as an MLflow run named ``rollback``. Only the alias moves
(no tag is changed), so a rollback can itself be undone with another rollback.

Redeploying is a separate step: the app resolves the alias again when it restarts
(Render deploy hook, see scripts/deploy_render.sh). Locally:
  python -m ml.rollback --model-name sms-spam --version 1
  RENDER_DEPLOY_HOOK_URL=... sh scripts/deploy_render.sh ghcr.io/<owner>/<repo>:sha-<commit>
  python scripts/smoke_test.py --url $PROD_URL --expected-version 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.lineage import git_sha
from ml.registry_utils import (
    CHAMPION_ALIAS,
    PREVIOUS_CHAMPION_TAG,
    STATUS_APPROVED,
    STATUS_TAG,
    get_client,
    get_version_by_alias,
    set_alias,
)

# Same experiment as the CT decisions: every alias change is visible in one place.
OPERATIONS_EXPERIMENT = "sms-spam-ct"

logger = logging.getLogger(__name__)


class RollbackError(RuntimeError):
    """The rollback is refused; nothing was changed."""


@dataclass
class RollbackResult:
    model_name: str
    from_version: str | None
    to_version: str
    changed: bool
    reason: str
    run_id: str | None = None


def resolve_target(client: MlflowClient, model_name: str, target: str | None) -> tuple:
    """Return (current champion version or None, target version), validated."""
    current = get_version_by_alias(client, model_name, CHAMPION_ALIAS)
    current_version = str(current.version) if current else None

    if not target:
        if current is None:
            raise RollbackError(f"{model_name} has no champion: give an explicit --version")
        target = current.tags.get(PREVIOUS_CHAMPION_TAG)
        if not target:
            raise RollbackError(
                f"champion version {current_version} has no {PREVIOUS_CHAMPION_TAG!r} tag: "
                "give an explicit --version"
            )
    target = str(target)

    try:
        mv = client.get_model_version(model_name, target)
    except MlflowException as exc:
        raise RollbackError(f"{model_name} version {target} does not exist") from exc
    status = mv.tags.get(STATUS_TAG)
    if status != STATUS_APPROVED:
        raise RollbackError(
            f"{model_name} version {target} has status {status!r}: only approved versions "
            "can become champion"
        )
    return current_version, target


def rollback(
    model_name: str,
    target: str | None = None,
    reason: str = "manual rollback",
    client: MlflowClient | None = None,
) -> RollbackResult:
    client = client or get_client()
    current, target = resolve_target(client, model_name, target)  # raises before any change
    changed = current != target
    if changed:
        set_alias(client, model_name, target, CHAMPION_ALIAS)
        logger.info("%s@%s: %s -> %s", model_name, CHAMPION_ALIAS, current, target)
    else:
        logger.info("%s@%s already points to %s: nothing to do", model_name, CHAMPION_ALIAS, target)

    result = RollbackResult(model_name, current, target, changed, reason)
    mlflow.set_experiment(OPERATIONS_EXPERIMENT)
    with mlflow.start_run(run_name="rollback") as run:
        mlflow.set_tags(
            {
                "operation": "rollback",
                "model_name": model_name,
                "champion_version_before": current or "none",
                "champion_version_after": target,
                "changed": str(changed).lower(),
                "reason": reason,
                "git_sha": git_sha(),
            }
        )
        result.run_id = run.info.run_id
    return result


def write_outputs(result: RollbackResult) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"from_version={result.from_version or ''}\n")
            fh.write(f"to_version={result.to_version}\n")
            fh.write(f"changed={str(result.changed).lower()}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(
                f"### Rollback {result.model_name}: {result.from_version or 'none'} -> "
                f"{result.to_version}\n\n- reason: {result.reason}\n"
                f"- alias changed: {'yes' if result.changed else 'no (already champion)'}\n"
                f"- MLflow run: {result.run_id}\n\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sms-spam")
    parser.add_argument("--version", help="target version (default: previous_champion tag)")
    parser.add_argument("--reason", default="manual rollback")
    args = parser.parse_args(argv)
    try:
        result = rollback(args.model_name, args.version, args.reason)
    except RollbackError as exc:
        logger.error("Rollback refused: %s", exc)
        return 1
    write_outputs(result)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
