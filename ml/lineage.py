"""Lineage helpers: Git commit and DVC hash of a data release."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_sha() -> str:
    """Commit being trained/served: GITHUB_SHA in Actions, else local HEAD."""
    return os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes to tracked files."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return bool(status)


def dvc_md5(dvc_file: Path) -> str:
    """md5 recorded by DVC for the (single) output of a .dvc file."""
    with Path(dvc_file).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["outs"][0]["md5"]
