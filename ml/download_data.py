"""Download the UCI SMS Spam Collection zip and extract it into data/raw/.

Usage: python -m ml.download_data [--dest data/raw]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

from ml.data_utils import RAW_DIR, RAW_FILENAME, sha256_file

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Only these members are extracted (never trust arbitrary paths inside an archive).
EXPECTED_MEMBERS = {RAW_FILENAME, "readme"}

logger = logging.getLogger(__name__)


def download(dest: Path, url: str = URL, timeout: int = 60) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        missing = EXPECTED_MEMBERS - names
        if missing:
            raise RuntimeError(f"Unexpected archive content, missing: {sorted(missing)}")
        for name in sorted(EXPECTED_MEMBERS):
            (dest / name).write_bytes(archive.read(name))

    raw_path = dest / RAW_FILENAME
    logger.info("Saved %s (sha256=%s)", raw_path, sha256_file(raw_path))
    return raw_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)
    try:
        download(args.dest)
    except Exception as exc:  # network errors, bad zip...
        logger.error("Download failed: %s", exc)
        logger.error("Manual fallback: download %s and unzip it into %s", URL, args.dest)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
