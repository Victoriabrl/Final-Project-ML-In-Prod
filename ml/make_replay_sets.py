"""Build reproducible replay sets from a DVC data release, for scripts/replay.py.

- ``normal.csv``: the release's held-out split, shuffled. Traffic that looks like the
  training data: the API's drift should stay below the threshold.
- ``campaign.csv``: a simulated spam campaign (spam messages only, part of them in
  capitals), sampled from the release's train and held-out splits. Longer messages, more
  digits, capitals, links and high spam scores: the drift should go above the threshold.

Same release + same seed = same files (sha256 in ``manifest.json``). The output folder
``.exercise-data/`` is ignored by Git and by the Docker image.

Usage (release fetched first with ``dvc pull data/releases/v2.dvc``):
  python -m ml.make_replay_sets [--data-version v2] [--out .exercise-data]
                                [--campaign-size 600] [--upper-ratio 0.5] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ml.data_utils import RELEASES_DIR, file_entry, read_csv, write_csv, write_json

logger = logging.getLogger(__name__)
DEFAULT_OUT = Path(".exercise-data")


def make_replay_sets(
    data_version: str = "v2",
    data_root: Path = RELEASES_DIR,
    out_dir: Path = DEFAULT_OUT,
    campaign_size: int = 600,
    upper_ratio: float = 0.5,
    seed: int = 42,
) -> dict:
    release_dir = Path(data_root) / data_version
    if not (release_dir / "holdout.csv").exists():
        raise FileNotFoundError(
            f"release {data_version!r} not found under {data_root} "
            f"(run: dvc pull {Path(data_root) / f'{data_version}.dvc'})"
        )
    train = read_csv(release_dir / "train.csv")
    holdout = read_csv(release_dir / "holdout.csv")

    normal = holdout.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    spam = pd.concat([train, holdout])
    spam = spam[spam["label"] == "spam"].drop_duplicates("text")
    campaign = spam.sample(n=campaign_size, replace=len(spam) < campaign_size, random_state=seed)
    campaign = campaign.reset_index(drop=True)
    shout = campaign.sample(frac=upper_ratio, random_state=seed).index
    campaign.loc[shout, "text"] = campaign.loc[shout, "text"].str.upper()

    out_dir = Path(out_dir)
    files = {}
    for name, df in (("normal.csv", normal), ("campaign.csv", campaign)):
        write_csv(df, out_dir / name)
        files[name] = file_entry(df, out_dir / name)
    manifest = {
        "data_version": data_version,
        "seed": seed,
        "campaign_size": campaign_size,
        "upper_ratio": upper_ratio,
        "files": files,
    }
    write_json(manifest, out_dir / "manifest.json")
    logger.info("replay sets written to %s: %s", out_dir, json.dumps(files))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", default="v2")
    parser.add_argument("--data-root", type=Path, default=RELEASES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--campaign-size", type=int, default=600)
    parser.add_argument("--upper-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        make_replay_sets(
            args.data_version, args.data_root, args.out, args.campaign_size, args.upper_ratio,
            args.seed,
        )  # fmt: skip
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
