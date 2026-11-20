"""Train a TF-IDF + logistic regression pipeline, log it to MLflow, optionally register it.

Usage:
  python -m ml.train --data-version v1 --params ml/config/params_baseline.yaml \
      --model-name sms-spam [--register]

MLflow location and credentials come from the environment (MLFLOW_TRACKING_URI, ...).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.data_utils import RELEASES_DIR, read_csv, sha256_file
from ml.evaluate import evaluate, save_reports
from ml.lineage import dvc_md5, git_dirty, git_sha
from ml.registry_utils import STATUS_CANDIDATE, get_client, set_version_tags
from ml.validate_data import validate_release

EXPERIMENT = "sms-spam-training"
logger = logging.getLogger(__name__)


def load_params(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_pipeline(params: dict) -> Pipeline:
    """One TF-IDF vectorizer per entry of params['features'] (all on the ``text`` column),
    concatenated, followed by a logistic regression."""
    transformers = []
    for name, cfg in params["features"].items():
        cfg = dict(cfg)
        cfg["ngram_range"] = tuple(cfg["ngram_range"])
        transformers.append((name, TfidfVectorizer(**cfg), "text"))
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("clf", LogisticRegression(**params["classifier"])),
        ]
    )


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{full}."))
        else:
            out[full] = str(value)
    return out


def train(
    data_version: str,
    params_path: Path,
    model_name: str,
    register: bool = False,
    data_root: Path = RELEASES_DIR,
    experiment: str = EXPERIMENT,
) -> dict:
    release_dir = data_root / data_version
    errors = validate_release(release_dir)
    if errors:
        raise ValueError(f"data contract failed for {release_dir}: {errors}")

    params = load_params(params_path)
    train_df = read_csv(release_dir / "train.csv")
    holdout_df = read_csv(release_dir / "holdout.csv")

    model = build_pipeline(params)
    model.fit(train_df[["text"]], train_df["label"])
    metrics, y_pred = evaluate(model, holdout_df)
    logger.info("%s on %s: %s", params.get("name"), data_version, metrics)

    sha = git_sha()
    dvc_file = data_root / f"{data_version}.dvc"
    lineage = {
        "git_sha": sha,
        "git_dirty": str(git_dirty()).lower(),
        "data_version": data_version,
        "dvc_md5": dvc_md5(dvc_file) if dvc_file.exists() else "unknown",
        "manifest_sha256": sha256_file(release_dir / "manifest.json"),
        "params_file": Path(params_path).as_posix(),
        "model_family": params.get("name", "unknown"),
    }

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{params.get('name', 'model')}-{data_version}") as run:
        mlflow.set_tags(lineage)
        mlflow.log_params(flatten({k: v for k, v in params.items() if k != "name"}))
        mlflow.log_params(
            {
                "data.train_rows": len(train_df),
                "data.train_spam": int((train_df["label"] == "spam").sum()),
                "data.holdout_rows": len(holdout_df),
            }
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            for path in save_reports(holdout_df["label"], y_pred, Path(tmp)):
                mlflow.log_artifact(str(path), artifact_path="evaluation")
        mlflow.log_artifact(str(release_dir / "manifest.json"), artifact_path="data")
        mlflow.log_artifact(str(params_path), artifact_path="config")

        example = holdout_df[["text"]].head(3)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            registered_model_name=model_name if register else None,
        )

        version = None
        if register:
            version = str(info.registered_model_version)
            set_version_tags(
                get_client(),
                model_name,
                version,
                {
                    "data_version": data_version,
                    "git_sha": sha,
                    "f1_spam": f"{metrics['f1_spam']:.4f}",
                    "status": STATUS_CANDIDATE,
                },
            )
            logger.info("Registered %s version %s (status=candidate)", model_name, version)

    return {
        "run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "model_name": model_name if register else None,
        "version": version,
        "metrics": metrics,
        "data_version": data_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--model-name", default="sms-spam")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--data-root", type=Path, default=RELEASES_DIR)
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args(argv)

    result = train(
        args.data_version,
        args.params,
        args.model_name,
        register=args.register,
        data_root=args.data_root,
        experiment=args.experiment,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
