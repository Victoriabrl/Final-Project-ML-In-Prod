"""Check that the configured MLflow registry supports model aliases.

Registers a tiny dummy model, sets an alias, reads it back with
get_model_version_by_alias and loads the model through the alias.
Run it against DagsHub before relying on aliases:

  MLFLOW_TRACKING_URI=... MLFLOW_TRACKING_USERNAME=... MLFLOW_TRACKING_PASSWORD=... \
      python -m scripts.check_registry_aliases
"""

from __future__ import annotations

import argparse
import sys

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from ml.evaluate import to_frame
from ml.train import build_pipeline

TINY_PARAMS = {
    "features": {"word": {"ngram_range": [1, 1]}},
    "classifier": {"C": 1.0, "max_iter": 200},
}
TEXTS = ["win free cash now", "call to claim prize", "see you at lunch", "ok thanks mum"]
LABELS = ["spam", "spam", "ham", "ham"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="alias-check")
    parser.add_argument("--alias", default="probe")
    args = parser.parse_args(argv)

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    model = build_pipeline(TINY_PARAMS).fit(to_frame(TEXTS), LABELS)

    mlflow.set_experiment("alias-check")
    with mlflow.start_run(run_name="alias-check"):
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=args.model_name)
    version = str(info.registered_model_version)
    print(f"Registered {args.model_name} version {version}")

    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, version)
    resolved = client.get_model_version_by_alias(args.model_name, args.alias)
    print(f"Alias {args.alias!r} -> version {resolved.version}")
    if str(resolved.version) != version:
        print("FAIL: alias resolved to the wrong version")
        return 1

    loaded = mlflow.sklearn.load_model(f"models:/{args.model_name}@{args.alias}")
    print(f"Prediction through alias: {loaded.predict(to_frame(['free prize, call now']))}")
    print("OK: aliases are supported by this registry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
