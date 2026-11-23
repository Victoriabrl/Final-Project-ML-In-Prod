"""Score a model on the held-out set of a data release.

Usage: python -m ml.evaluate --model-uri models:/sms-spam@champion --data-version v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ml.data_utils import RELEASES_DIR, read_csv

POSITIVE = "spam"
LABEL_ORDER = ["ham", "spam"]


def to_frame(texts: Iterable[str]) -> pd.DataFrame:
    """Model input: a DataFrame with a single ``text`` column."""
    return pd.DataFrame({"text": list(texts)})


def spam_probability(model, texts: Iterable[str]) -> np.ndarray:
    proba = model.predict_proba(to_frame(texts))
    return proba[:, list(model.classes_).index(POSITIVE)]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    kw = {"pos_label": POSITIVE, "zero_division": 0}
    return {
        "f1_spam": float(f1_score(y_true, y_pred, **kw)),
        "precision_spam": float(precision_score(y_true, y_pred, **kw)),
        "recall_spam": float(recall_score(y_true, y_pred, **kw)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate(model, df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    y_pred = np.asarray(model.predict(df[["text"]]))
    return compute_metrics(df["label"], y_pred), y_pred


def save_reports(y_true, y_pred, out_dir: Path) -> list[Path]:
    """Write confusion_matrix.png and classification_report.txt into out_dir."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABEL_ORDER, cmap="Blues", colorbar=False
    )
    disp.ax_.set_title("Held-out confusion matrix")
    disp.figure_.tight_layout()
    disp.figure_.savefig(cm_path, dpi=120)
    plt.close(disp.figure_)

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(
        classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0),
        encoding="utf-8",
    )
    return [cm_path, report_path]


def load_holdout(data_version: str, root: Path = RELEASES_DIR) -> pd.DataFrame:
    return read_csv(root / data_version / "holdout.csv")


def main(argv: list[str] | None = None) -> int:
    import mlflow.sklearn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--data-root", type=Path, default=RELEASES_DIR)
    args = parser.parse_args(argv)

    model = mlflow.sklearn.load_model(args.model_uri)
    metrics, _ = evaluate(model, load_holdout(args.data_version, args.data_root))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
