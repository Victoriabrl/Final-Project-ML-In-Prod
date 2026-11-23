"""Create a small MLflow registry for tests and the CI container smoke test.

Two tiny pipelines trained on a dozen messages are registered under one model name:
  - version 1: status=approved, alias ``champion``;
  - version 2: newer, status=candidate, no alias.

Usage:
  python -m scripts.make_test_registry DIR                 # sqlite registry in DIR
  python -m scripts.make_test_registry --tracking-uri URI  # e.g. a local `mlflow server`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from ml.train import build_pipeline

TRAIN_TEXTS = [
    "WINNER! You have won a free prize, call now to claim",
    "Free entry to win cash, text WIN to 80086 now",
    "URGENT! Your mobile won a 2000 cash award, call 09061701461",
    "Claim your free voucher now, reply YES to win",
    "Congratulations, you won a free holiday, call to claim your prize",
    "Free ringtone offer, txt STOP to cancel, win cash prize",
    "Are we still meeting for lunch tomorrow?",
    "Ok see you at home later, love you",
    "Can you pick up some bread on the way back?",
    "I will be late for the meeting, sorry",
    "Thanks for dinner yesterday, it was great",
    "Call me when you get home please",
]
TRAIN_LABELS = ["spam"] * 6 + ["ham"] * 6

TINY_PARAMS = {
    "v1": {
        "features": {"word": {"ngram_range": [1, 1]}},
        "classifier": {"C": 10.0, "class_weight": "balanced", "max_iter": 500},
    },
    "v2": {
        "features": {"word": {"ngram_range": [1, 2]}},
        "classifier": {"C": 5.0, "class_weight": "balanced", "max_iter": 500},
    },
}


def build_registry(
    tracking_uri: str, artifact_location: str | None = None, model_name: str = "sms-spam"
) -> dict:
    import pandas as pd

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp_name = "test-registry"
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else client.create_experiment(exp_name, artifact_location)

    X = pd.DataFrame({"text": TRAIN_TEXTS})
    versions = {}
    for data_version, params in TINY_PARAMS.items():
        model = build_pipeline(params).fit(X, TRAIN_LABELS)
        with mlflow.start_run(experiment_id=exp_id, run_name=f"tiny-{data_version}"):
            info = mlflow.sklearn.log_model(
                model, name="model", registered_model_name=model_name, input_example=X.head(2)
            )
        version = str(info.registered_model_version)
        status = "approved" if data_version == "v1" else "candidate"
        client.set_model_version_tag(model_name, version, "status", status)
        client.set_model_version_tag(model_name, version, "data_version", data_version)
        versions[data_version] = version

    client.set_registered_model_alias(model_name, "champion", versions["v1"])
    return {
        "tracking_uri": tracking_uri,
        "model_name": model_name,
        "champion_version": versions["v1"],
        "newer_version": versions["v2"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", nargs="?", type=Path, help="directory for a sqlite registry")
    parser.add_argument("--tracking-uri", help="use an existing tracking server instead")
    parser.add_argument("--model-name", default="sms-spam")
    args = parser.parse_args(argv)

    if args.tracking_uri:
        result = build_registry(args.tracking_uri, model_name=args.model_name)
    elif args.dir:
        root = args.dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        result = build_registry(
            f"sqlite:///{(root / 'mlflow.db').as_posix()}",
            artifact_location=(root / "artifacts").as_uri(),
            model_name=args.model_name,
        )
    else:
        parser.error("give a DIR or --tracking-uri")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
