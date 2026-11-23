"""Data contract for a dataset release. Exits non-zero with clear messages on failure.

Usage: python -m ml.validate_data --version v1 [--root data/releases]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ml.data_utils import COLUMNS, LABELS, RELEASES_DIR, read_csv, read_json, sha256_file

MIN_SPAM_RATE = 0.05
MAX_SPAM_RATE = 0.30
FILES = ("train.csv", "holdout.csv")

logger = logging.getLogger(__name__)


def check_frame(df: pd.DataFrame, name: str) -> list[str]:
    """Contract checks for one CSV file."""
    if list(df.columns) != COLUMNS:
        return [f"{name}: columns must be exactly {COLUMNS}, got {list(df.columns)}"]

    errors = []
    if df.empty:
        return [f"{name}: file is empty"]
    if df.isna().any().any():
        errors.append(f"{name}: contains null values")

    unknown = sorted(set(df["label"].dropna()) - LABELS)
    if unknown:
        errors.append(f"{name}: unknown labels {unknown} (allowed: {sorted(LABELS)})")

    empty = int((df["text"].fillna("").str.strip() == "").sum())
    if empty:
        errors.append(f"{name}: {empty} empty text(s)")

    spam_rate = float((df["label"] == "spam").mean())
    if not MIN_SPAM_RATE <= spam_rate <= MAX_SPAM_RATE:
        errors.append(
            f"{name}: spam rate {spam_rate:.1%} outside [{MIN_SPAM_RATE:.0%}, {MAX_SPAM_RATE:.0%}]"
        )
    return errors


def check_leakage(train: pd.DataFrame, holdout: pd.DataFrame) -> list[str]:
    """Train and held-out must not share any text."""
    if "text" not in train.columns or "text" not in holdout.columns:
        return []
    overlap = set(train["text"].str.strip()) & set(holdout["text"].str.strip())
    if overlap:
        return [f"leakage: {len(overlap)} text(s) present in both train and holdout"]
    return []


def check_manifest(release_dir: Path, frames: dict[str, pd.DataFrame]) -> list[str]:
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = read_json(manifest_path)
    errors = []
    if manifest.get("version") != release_dir.name:
        errors.append(
            f"manifest version {manifest.get('version')!r} != directory {release_dir.name!r}"
        )
    for name in FILES:
        entry = manifest.get("files", {}).get(name)
        if entry is None:
            errors.append(f"manifest: no entry for {name}")
            continue
        actual = sha256_file(release_dir / name)
        if actual != entry.get("sha256"):
            errors.append(f"{name}: sha256 mismatch (manifest {entry.get('sha256')}, got {actual})")
        if name in frames and entry.get("rows") != len(frames[name]):
            errors.append(f"{name}: {len(frames[name])} rows, manifest says {entry.get('rows')}")
    return errors


def validate_release(release_dir: Path) -> list[str]:
    """Return the list of contract violations (empty list = valid)."""
    release_dir = Path(release_dir)
    if not release_dir.is_dir():
        return [f"release directory not found: {release_dir}"]

    errors: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for name in FILES:
        path = release_dir / name
        if not path.exists():
            errors.append(f"missing file {path}")
            continue
        frames[name] = read_csv(path)
        errors += check_frame(frames[name], name)

    if len(frames) == len(FILES):
        errors += check_leakage(frames["train.csv"], frames["holdout.csv"])
    errors += check_manifest(release_dir, frames)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="release name, e.g. v1")
    parser.add_argument("--root", type=Path, default=RELEASES_DIR)
    args = parser.parse_args(argv)

    release_dir = args.root / args.version
    errors = validate_release(release_dir)
    if errors:
        logger.error("Data contract FAILED for %s:", release_dir)
        for err in errors:
            logger.error("  - %s", err)
        return 1
    logger.info("Data contract OK for %s", release_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
