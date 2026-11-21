"""Data contract tests on small hand-made releases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data_utils import file_entry, write_csv, write_json
from ml.make_splits import clean
from ml.validate_data import validate_release

HAM = [f"hello friend number {i}, see you at lunch" for i in range(8)]
SPAM = ["WIN a FREE prize now, call 0800", "URGENT! claim your cash reward txt WIN"]
HOLDOUT_HAM = [f"are we still meeting at {h}pm today?" for h in range(8)]
HOLDOUT_SPAM = ["Congratulations you won a voucher reply YES", "Free entry in a weekly draw txt"]


def frame(ham: list[str], spam: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"label": ["ham"] * len(ham) + ["spam"] * len(spam), "text": ham + spam})


def make_release(root: Path, train: pd.DataFrame, holdout: pd.DataFrame, version="v1") -> Path:
    """Write CSVs then a manifest consistent with them."""
    out = root / version
    out.mkdir(parents=True)
    files = {}
    for name, df in (("train.csv", train), ("holdout.csv", holdout)):
        write_csv(df, out / name)
        files[name] = file_entry(df, out / name)
    write_json({"version": version, "seed": 42, "files": files}, out / "manifest.json")
    return out


@pytest.fixture
def train_df() -> pd.DataFrame:
    return frame(HAM, SPAM)


@pytest.fixture
def holdout_df() -> pd.DataFrame:
    return frame(HOLDOUT_HAM, HOLDOUT_SPAM)


def test_valid_release_passes(tmp_path, train_df, holdout_df):
    assert validate_release(make_release(tmp_path, train_df, holdout_df)) == []


def test_unknown_label_detected(tmp_path, train_df, holdout_df):
    train_df.loc[0, "label"] = "promo"
    errors = validate_release(make_release(tmp_path, train_df, holdout_df))
    assert any("unknown labels" in e for e in errors)


@pytest.mark.parametrize("bad_text", ["", "   "])
def test_empty_text_detected(tmp_path, train_df, holdout_df, bad_text):
    train_df.loc[0, "text"] = bad_text
    errors = validate_release(make_release(tmp_path, train_df, holdout_df))
    assert any("empty text" in e for e in errors)


def test_missing_column_detected(tmp_path, train_df, holdout_df):
    out = make_release(tmp_path, train_df, holdout_df)
    train_df[["text"]].to_csv(out / "train.csv", index=False)
    errors = validate_release(out)
    assert any("columns must be exactly" in e for e in errors)


def test_extra_column_detected(tmp_path, train_df, holdout_df):
    out = make_release(tmp_path, train_df, holdout_df)
    train_df.assign(extra=1).to_csv(out / "train.csv", index=False)
    assert any("columns must be exactly" in e for e in validate_release(out))


def test_train_holdout_leakage_detected(tmp_path, train_df, holdout_df):
    holdout_df.loc[0, "text"] = train_df.loc[0, "text"]
    errors = validate_release(make_release(tmp_path, train_df, holdout_df))
    assert any("leakage" in e for e in errors)


def test_spam_rate_out_of_bounds_detected(tmp_path, holdout_df):
    no_spam = frame(HAM, [])
    errors = validate_release(make_release(tmp_path, no_spam, holdout_df))
    assert any("spam rate" in e for e in errors)


def test_hash_mismatch_detected(tmp_path, train_df, holdout_df):
    out = make_release(tmp_path, train_df, holdout_df)
    with (out / "holdout.csv").open("a", encoding="utf-8") as fh:
        fh.write("ham,a message added after the manifest\n")
    errors = validate_release(out)
    assert any("sha256 mismatch" in e for e in errors)


def test_missing_release_dir(tmp_path):
    assert validate_release(tmp_path / "v9") != []


def test_clean_removes_duplicates_and_conflicts():
    raw = pd.DataFrame(
        {
            "label": ["ham", "ham", "spam", "ham", "spam", "ham"],
            "text": ["hi there ", "hi there", "free cash", "free cash", "win now", "  "],
        }
    )
    df, stats = clean(raw)
    # "hi there" deduplicated, "free cash" dropped (contradictory), blank text dropped.
    assert sorted(df["text"]) == ["hi there", "win now"]
    assert stats["conflicting_texts"] == 1
    assert stats["duplicate_rows_removed"] == 1
    assert stats["empty_text_removed"] == 1
