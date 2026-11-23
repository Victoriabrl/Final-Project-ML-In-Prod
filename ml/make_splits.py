"""Build deterministic dataset releases from the raw UCI file.

Steps (seed 42, fully deterministic):
  a) strip surrounding whitespace, drop empty texts, drop texts that appear with
     contradictory labels, drop exact duplicate texts;
  b) stratified 20 % held-out set, identical (same bytes, same hash) in every release;
  c) shuffle the remaining pool and cut it into batches simulating new labelled data;
  d) v1 train = first batch (800 messages);
  e) v2 train = whole pool (v1 + following batches);
  f) each release gets a manifest.json (counts, SHA-256 of each CSV, change description).

Usage: python -m ml.make_splits --release v1   (or v2, or all)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data_utils import (
    RAW_DIR,
    RAW_FILENAME,
    RELEASES_DIR,
    file_entry,
    read_raw,
    sha256_file,
    write_csv,
    write_json,
)

SEED = 42
HOLDOUT_FRACTION = 0.20
V1_SIZE = 800
V1_MIN_SPAM = 80
SOURCE = "UCI SMS Spam Collection (id 228), https://doi.org/10.24432/C5CC84, CC BY 4.0"

DESCRIPTIONS = {
    "v1": (
        f"Initial release: train = first {V1_SIZE} messages of the shuffled pool "
        "(first batch of labelled data)."
    ),
    "v2": (
        "New labelled messages arrived: train = whole pool (v1 batch + all following "
        "batches). Held-out set unchanged."
    ),
}

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise and deduplicate; return the clean frame and the counts removed."""
    stats = {"raw_rows": int(len(df))}
    df = df.assign(text=df["text"].str.strip(), label=df["label"].str.strip())

    empty = df["text"] == ""
    stats["empty_text_removed"] = int(empty.sum())
    df = df[~empty]

    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    conflict_mask = df["text"].isin(conflicting)
    stats["conflicting_texts"] = len(conflicting)
    stats["conflicting_rows_removed"] = int(conflict_mask.sum())
    df = df[~conflict_mask]

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    stats["duplicate_rows_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    stats["clean_rows"] = int(len(df))
    stats["clean_spam"] = int((df["label"] == "spam").sum())
    return df, stats


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (holdout, shuffled pool)."""
    pool, holdout = train_test_split(
        df, test_size=HOLDOUT_FRACTION, stratify=df["label"], random_state=SEED
    )
    holdout = holdout.sort_index().reset_index(drop=True)
    pool = pool.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return holdout, pool


def release_train(pool: pd.DataFrame, version: str) -> pd.DataFrame:
    if version == "v1":
        train = pool.iloc[:V1_SIZE]
        n_spam = int((train["label"] == "spam").sum())
        if n_spam < V1_MIN_SPAM:
            raise ValueError(f"v1 has only {n_spam} spams (< {V1_MIN_SPAM})")
        return train
    if version == "v2":
        return pool
    raise ValueError(f"Unknown release {version!r}")


def build_release(
    version: str,
    holdout: pd.DataFrame,
    pool: pd.DataFrame,
    out_root: Path,
    cleaning_stats: dict,
    raw_sha256: str,
) -> Path:
    out_dir = out_root / version
    out_dir.mkdir(parents=True, exist_ok=True)
    train = release_train(pool, version)

    train_path, holdout_path = out_dir / "train.csv", out_dir / "holdout.csv"
    write_csv(train, train_path)
    write_csv(holdout, holdout_path)

    manifest = {
        "version": version,
        "seed": SEED,
        "source": SOURCE,
        "raw_sha256": raw_sha256,
        "cleaning": cleaning_stats,
        "split": {
            "holdout_fraction": HOLDOUT_FRACTION,
            "pool_rows": int(len(pool)),
            "v1_size": V1_SIZE,
        },
        "files": {
            "train.csv": file_entry(train, train_path),
            "holdout.csv": file_entry(holdout, holdout_path),
        },
        "description": DESCRIPTIONS[version],
    }
    write_json(manifest, out_dir / "manifest.json")
    files = manifest["files"]
    logger.info(
        "%s: train=%d rows (%d spam), holdout=%d rows (%d spam) -> %s",
        version,
        files["train.csv"]["rows"],
        files["train.csv"]["spam"],
        files["holdout.csv"]["rows"],
        files["holdout.csv"]["spam"],
        out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=["v1", "v2", "all"], required=True)
    parser.add_argument("--raw", type=Path, default=RAW_DIR / RAW_FILENAME)
    parser.add_argument("--out", type=Path, default=RELEASES_DIR)
    args = parser.parse_args(argv)

    raw = read_raw(args.raw)
    df, stats = clean(raw)
    logger.info("Cleaning: %s", stats)
    holdout, pool = split(df)

    versions = ["v1", "v2"] if args.release == "all" else [args.release]
    raw_sha256 = sha256_file(args.raw)
    for version in versions:
        build_release(version, holdout, pool, args.out, stats, raw_sha256)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
