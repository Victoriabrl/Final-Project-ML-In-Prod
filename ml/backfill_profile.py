"""Add the training profile to a registered version trained before profiles existed.

The profile is rebuilt exactly as ``ml/train.py`` now does it (text statistics of the
release's training split, the model's spam probabilities on its held-out split) and
logged as ``monitoring/training-profile.json`` on the version's own training run. The
model itself is not retrained or modified.

Usage (release fetched first with ``dvc pull data/releases/<version>.dvc``):
  python -m ml.backfill_profile --model-name sms-spam --version 5 [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow.sklearn
from mlflow import MlflowClient

from app.drift import PROFILE_ARTIFACT, build_profile
from ml.data_utils import RELEASES_DIR, read_csv
from ml.evaluate import spam_probability
from ml.registry_utils import get_client

logger = logging.getLogger(__name__)


def has_profile(client: MlflowClient, run_id: str) -> bool:
    folder = PROFILE_ARTIFACT.rsplit("/", 1)[0]
    return any(a.path == PROFILE_ARTIFACT for a in client.list_artifacts(run_id, folder))


def backfill(
    model_name: str,
    version: str,
    client: MlflowClient | None = None,
    data_root: Path = RELEASES_DIR,
    force: bool = False,
) -> dict:
    client = client or get_client()
    mv = client.get_model_version(model_name, version)
    if not force and has_profile(client, mv.run_id):
        logger.info("run %s already has %s: nothing to do", mv.run_id, PROFILE_ARTIFACT)
        return {"run_id": mv.run_id, "written": False}

    data_version = mv.tags.get("data_version")
    release_dir = Path(data_root) / str(data_version)
    if not data_version or not (release_dir / "train.csv").exists():
        raise FileNotFoundError(
            f"release {data_version!r} not found under {data_root} "
            f"(run: dvc pull {Path(data_root) / f'{data_version}.dvc'})"
        )

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    train_df = read_csv(release_dir / "train.csv")
    holdout_df = read_csv(release_dir / "holdout.csv")
    profile = build_profile(
        train_df["text"],
        spam_probability(model, holdout_df["text"]),
        source={"data_version": data_version, "backfilled_for_version": str(version)},
    )
    client.log_dict(mv.run_id, profile, PROFILE_ARTIFACT)
    client.set_tag(mv.run_id, "profile_backfilled", "true")
    logger.info("logged %s on run %s (v%s)", PROFILE_ARTIFACT, mv.run_id, version)
    return {"run_id": mv.run_id, "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sms-spam")
    parser.add_argument("--version", required=True)
    parser.add_argument("--data-root", type=Path, default=RELEASES_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite an existing profile")
    args = parser.parse_args(argv)
    try:
        result = backfill(args.model_name, args.version, data_root=args.data_root, force=args.force)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
