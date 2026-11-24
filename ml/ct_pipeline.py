"""Continuous-training (CT) pipeline, called by .github/workflows/ct.yml.

1. Fetch (optional ``dvc pull``) and validate the requested release. Failure: stop,
   nothing is changed in the registry or in production.
2. Score the current champion (if any) on the release held-out set and apply
   ``should_retrain``. No reason and no ``--force-train``: the champion is kept.
3. Train a candidate (baseline when there is no champion yet = bootstrap, candidate
   params otherwise), register it (status=candidate) and compare with ``decide``
   on the same held-out observations.
4. Promoted: status=approved, alias moved, previous_champion tag; with ``--deploy``,
   redeploy production and check it serves the new version. If it does not, the alias
   goes back to the previous champion, production is redeployed and checked, and the
   pipeline fails.
5. Rejected: the champion stays, the candidate gets status=rejected + reason.
6. Always: a parent MLflow run ``ct-decision`` logs data version, DVC hash, Git SHA,
   scores, thresholds, decision, reason and champion versions before/after. When the
   champion has a training profile, ``drift.json`` compares the release with it (same
   PSI and threshold as the API ``feature_drift`` gauge). Drift is logged for
   investigation only: it never triggers retraining by itself.

Usage:
  python -m ml.ct_pipeline --data-version v2 [--model-name sms-spam] [--force-train]
                           [--deploy] [--pull]
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import subprocess
import sys
import urllib.error
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import mlflow
import mlflow.artifacts
import mlflow.sklearn
from mlflow import MlflowClient

from app.drift import PROFILE_ARTIFACT, drift_report
from ml.data_utils import RELEASES_DIR, read_csv, sha256_file
from ml.evaluate import evaluate, spam_probability
from ml.lineage import dvc_md5, git_sha
from ml.promotion import DEFAULT_CONFIG_PATH, decide, load_config, should_retrain
from ml.registry_utils import (
    CHAMPION_ALIAS,
    STATUS_REJECTED,
    get_client,
    get_version_by_alias,
    promote,
    set_alias,
    set_status,
)
from ml.train import train
from ml.validate_data import validate_release

CT_EXPERIMENT = "sms-spam-ct"
PROD_MODEL_NAME = "sms-spam"
PARAMS_BASELINE = Path("ml/config/params_baseline.yaml")
PARAMS_CANDIDATE = Path("ml/config/params_candidate.yaml")

logger = logging.getLogger(__name__)


class CTError(RuntimeError):
    """The pipeline must stop and the workflow must fail."""


class Deployer(Protocol):
    def deploy(self) -> None: ...

    def verify(self, expected_version: str) -> bool: ...


@dataclass
class RenderDeployer:
    """Redeploy the production image through the Render hook, then smoke test the URL.

    A restart is enough: the app resolves the champion alias again at startup.
    """

    hook_url: str
    prod_url: str
    image: str
    timeout: float = 900

    @classmethod
    def from_env(cls) -> RenderDeployer:
        names = ("RENDER_DEPLOY_HOOK_URL", "PROD_URL", "IMAGE")
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            raise CTError(f"deploy requested but not configured: missing {', '.join(missing)}")
        return cls(
            os.environ["RENDER_DEPLOY_HOOK_URL"], os.environ["PROD_URL"], os.environ["IMAGE"]
        )

    def deploy(self) -> None:
        env = {**os.environ, "RENDER_DEPLOY_HOOK_URL": self.hook_url}
        try:
            subprocess.run(["sh", "scripts/deploy_render.sh", self.image], env=env, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CTError(f"Render deploy hook failed: {exc}") from exc

    def verify(self, expected_version: str) -> bool:
        from scripts.smoke_test import run

        try:
            run(self.prod_url, expected_version, self.timeout)
        except (
            AssertionError,
            KeyError,
            OSError,
            urllib.error.URLError,
            http.client.HTTPException,
        ):
            logger.exception("Production check failed for version %s", expected_version)
            return False
        return True


@dataclass
class CTResult:
    decision: str  # promoted | rejected | kept
    reason: str
    data_version: str
    model_name: str
    champion_before: str | None = None
    champion_after: str | None = None
    champion_score: float | None = None
    candidate_version: str | None = None
    candidate_score: float | None = None
    retrain_reasons: list[str] = field(default_factory=list)
    deployed: bool = False
    run_id: str | None = None
    drift_status: str | None = None  # ok | drift | no_profile (None: no champion)
    drift_max_psi: float | None = None


def dvc_pull(dvc_file: Path) -> None:
    try:
        subprocess.run(["dvc", "pull", str(dvc_file)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CTError(f"dvc pull {dvc_file} failed: {exc}") from exc


def run_ct(
    data_version: str,
    model_name: str = PROD_MODEL_NAME,
    force_train: bool = False,
    deploy: bool = False,
    deployer: Deployer | None = None,
    client: MlflowClient | None = None,
    data_root: Path = RELEASES_DIR,
    promotion_config: Path = DEFAULT_CONFIG_PATH,
    params_baseline: Path = PARAMS_BASELINE,
    params_candidate: Path = PARAMS_CANDIDATE,
    pull: bool = False,
) -> CTResult:
    release_dir = Path(data_root) / data_version
    dvc_file = Path(data_root) / f"{data_version}.dvc"

    # 1. Fetch + validate before touching the registry.
    if pull:
        dvc_pull(dvc_file)
    errors = validate_release(release_dir)
    if errors:
        raise CTError(f"data contract failed for {release_dir}: " + "; ".join(errors))

    if deploy and model_name != PROD_MODEL_NAME:
        logger.info("deploy forced to false: %s is not the production model", model_name)
        deploy = False
    if deploy and deployer is None:
        raise CTError("deploy requested without a deployer")

    cfg = load_config(promotion_config)
    client = client or get_client()
    holdout = read_csv(release_dir / "holdout.csv")
    train_texts = read_csv(release_dir / "train.csv")["text"].tolist()
    result = CTResult(decision="", reason="", data_version=data_version, model_name=model_name)

    mlflow.set_experiment(CT_EXPERIMENT)
    with mlflow.start_run(run_name="ct-decision") as parent:
        result.run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "data_version": data_version,
                "dvc_md5": dvc_md5(dvc_file) if dvc_file.exists() else "unknown",
                "manifest_sha256": sha256_file(release_dir / "manifest.json"),
                "git_sha": git_sha(),
                "model_name": model_name,
                "force_train": str(force_train).lower(),
                "deploy": str(deploy).lower(),
            }
        )
        mlflow.log_params(
            {
                "metric": cfg.metric,
                "direction": cfg.direction,
                "quality_floor": cfg.quality_floor,
                "min_improvement": cfg.min_improvement,
            }
        )
        try:
            _decide_and_apply(
                result, cfg, client, holdout, train_texts, force_train, deploy, deployer,
                data_root, params_baseline, params_candidate,
            )  # fmt: skip
        except Exception as exc:
            current = get_version_by_alias(client, model_name, CHAMPION_ALIAS)
            result.champion_after = str(current.version) if current else None
            result.decision, result.reason = "failed", str(exc)
            _log_decision(result, cfg.metric)
            exc.ct_result = result  # lets main() write a complete summary
            raise
        _log_decision(result, cfg.metric)
    return result


def _decide_and_apply(
    result: CTResult,
    cfg,
    client: MlflowClient,
    holdout,
    train_texts: list[str],
    force_train: bool,
    deploy: bool,
    deployer: Deployer | None,
    data_root: Path,
    params_baseline: Path,
    params_candidate: Path,
) -> None:
    name = result.model_name

    # 2. Score the champion on this release's held-out set.
    champion = get_version_by_alias(client, name, CHAMPION_ALIAS)
    champion_data_version = None
    if champion is not None:
        result.champion_before = str(champion.version)
        champion_data_version = champion.tags.get("data_version")
        model = mlflow.sklearn.load_model(f"models:/{name}/{result.champion_before}")
        metrics, _ = evaluate(model, holdout)
        result.champion_score = metrics[cfg.metric]
        _log_drift(result, champion, model, train_texts, holdout)
    result.champion_after = result.champion_before

    retrain = should_retrain(result.data_version, champion_data_version, result.champion_score, cfg)
    result.retrain_reasons = list(retrain.reasons)
    if not retrain.retrain and not force_train:
        result.decision = "kept"
        result.reason = (
            f"champion kept: no reason to retrain (data {result.data_version} already used, "
            f"champion {cfg.metric}={result.champion_score:.4f} above the floor)"
        )
        return

    # 3. Train and register a candidate, compare on the same held-out observations.
    params = params_candidate if champion is not None else params_baseline
    trained = train(
        result.data_version, params, name, register=True, data_root=data_root, nested=True
    )
    result.candidate_version = trained["version"]
    result.candidate_score = trained["metrics"][cfg.metric]
    decision = decide(result.candidate_score, result.champion_score, cfg)

    if not decision.promote:
        # 5. Rejected: the champion stays in place.
        set_status(client, name, result.candidate_version, STATUS_REJECTED, decision.reason)
        result.decision, result.reason = "rejected", decision.reason
        return

    # 4. Promote, then deploy and verify production.
    previous = promote(client, name, result.candidate_version)
    result.champion_after = result.candidate_version
    result.decision, result.reason = "promoted", decision.reason
    if not deploy:
        return

    deployer.deploy()
    if deployer.verify(result.candidate_version):
        result.deployed = True
        return

    # Production does not serve the new version: restore the previous champion.
    reason = "production did not serve the new version after promotion"
    if previous:
        set_alias(client, name, previous)
    else:
        client.delete_registered_model_alias(name, CHAMPION_ALIAS)
    set_status(client, name, result.candidate_version, STATUS_REJECTED, reason)
    result.champion_after = previous
    restored = False
    if previous:
        deployer.deploy()
        restored = deployer.verify(previous)
    raise CTError(
        f"{reason} (version {result.candidate_version}); alias restored to "
        f"{previous or 'none'}, previous version serving again: {restored}"
    )


def _log_drift(result: CTResult, champion, model, train_texts: list[str], holdout) -> None:
    """Compare the release with the champion's training profile (drift.json)."""
    try:
        profile = mlflow.artifacts.load_dict(f"runs:/{champion.run_id}/{PROFILE_ARTIFACT}")
    except Exception as exc:  # champion trained before profiles existed, or no artifact
        logger.warning("no training profile for champion v%s: %s", champion.version, exc)
        result.drift_status = "no_profile"
        return
    report = drift_report(profile, train_texts, spam_probability(model, holdout["text"]))
    report["champion_version"] = str(champion.version)
    report["data_version"] = result.data_version
    mlflow.log_dict(report, "drift.json")
    mlflow.log_metric("drift_max_psi", report["max_psi"])
    result.drift_max_psi = report["max_psi"]
    result.drift_status = "drift" if report["drifted"] else "ok"
    if report["drifted"]:
        logger.warning("input drift vs champion profile: %s (investigate)", report["drifted"])


def _log_decision(result: CTResult, metric: str) -> None:
    mlflow.set_tags(
        {
            "decision": result.decision,
            "reason": result.reason[:5000],
            "retrain_reasons": ",".join(result.retrain_reasons) or "none",
            "champion_version_before": result.champion_before or "none",
            "champion_version_after": result.champion_after or "none",
            "candidate_version": result.candidate_version or "none",
            "deployed": str(result.deployed).lower(),
            "drift_status": result.drift_status or "none",
        }
    )
    if result.champion_score is not None:
        mlflow.log_metric(f"champion_{metric}", result.champion_score)
    if result.candidate_score is not None:
        mlflow.log_metric(f"candidate_{metric}", result.candidate_score)


def write_summary(result: CTResult, path: str | None) -> None:
    if not path:
        return
    icons = {"promoted": "🚀", "rejected": "🛑", "kept": "🟰", "failed": "❌"}

    def fmt(x):
        return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

    rows = [
        ("Data version", result.data_version),
        ("Model", result.model_name),
        ("Retrain reasons", ", ".join(result.retrain_reasons) or "none"),
        ("Champion before", fmt(result.champion_before)),
        ("Champion score", fmt(result.champion_score)),
        ("Candidate version", fmt(result.candidate_version)),
        ("Candidate score", fmt(result.candidate_score)),
        ("Champion after", fmt(result.champion_after)),
        ("Deployed", "yes" if result.deployed else "no"),
        ("Drift (max PSI)", f"{fmt(result.drift_max_psi)} ({result.drift_status or 'none'})"),
        ("MLflow run", fmt(result.run_id)),
    ]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"### CT {icons.get(result.decision, '')} {result.decision}\n\n")
        fh.write(f"{result.reason}\n\n| | |\n|---|---|\n")
        fh.writelines(f"| {k} | {v} |\n" for k, v in rows)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--model-name", default=PROD_MODEL_NAME)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--pull", action="store_true", help="dvc pull the release first")
    parser.add_argument("--data-root", type=Path, default=RELEASES_DIR)
    parser.add_argument("--promotion-config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    try:
        deploy = args.deploy and args.model_name == PROD_MODEL_NAME
        deployer = RenderDeployer.from_env() if deploy else None
        result = run_ct(
            args.data_version,
            model_name=args.model_name,
            force_train=args.force_train,
            deploy=deploy,
            deployer=deployer,
            data_root=args.data_root,
            promotion_config=args.promotion_config,
            pull=args.pull,
        )
    except CTError as exc:
        logger.error("CT failed: %s", exc)
        failed = getattr(exc, "ct_result", None)
        write_summary(
            failed or CTResult("failed", str(exc), args.data_version, args.model_name), summary
        )
        return 1
    write_summary(result, summary)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
