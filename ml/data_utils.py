"""Shared helpers for reading/writing dataset files and computing hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["label", "text"]
LABELS = {"ham", "spam"}
RAW_FILENAME = "SMSSpamCollection"
RELEASES_DIR = Path("data/releases")
RAW_DIR = Path("data/raw")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    """Read the raw UCI file: one message per line, ``label<TAB>text``.

    Parsed by hand (split on the first tab) so quotes are never interpreted.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            label, sep, text = line.partition("\t")
            if not sep:
                raise ValueError(f"{path}:{line_no}: missing tab separator")
            rows.append((label, text))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a ``label,text`` CSV deterministically (UTF-8, LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a release CSV keeping every value as a string (no NA inference)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def file_entry(df: pd.DataFrame, path: Path) -> dict:
    """Manifest entry for one CSV file."""
    return {
        "rows": int(len(df)),
        "spam": int((df["label"] == "spam").sum()),
        "sha256": sha256_file(path),
    }


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
