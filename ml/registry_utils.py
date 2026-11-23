"""MLflow Model Registry helpers: aliases, version tags and statuses."""

from __future__ import annotations

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

CHAMPION_ALIAS = "champion"
STATUS_TAG = "status"
PREVIOUS_CHAMPION_TAG = "previous_champion"

STATUS_CANDIDATE = "candidate"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

# Error codes MLflow uses when a registered model or an alias does not exist.
_NOT_FOUND_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"}


def get_client() -> MlflowClient:
    """Client configured from MLFLOW_TRACKING_URI / credentials in the environment."""
    return MlflowClient()


def get_version_by_alias(
    client: MlflowClient, model_name: str, alias: str = CHAMPION_ALIAS
) -> ModelVersion | None:
    """Model version behind an alias, or None if the model/alias does not exist.

    Any other error (network, auth...) is raised.
    """
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        if exc.error_code in _NOT_FOUND_CODES:
            return None
        raise


def set_version_tags(client: MlflowClient, model_name: str, version: str, tags: dict) -> None:
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, str(value))


def set_status(
    client: MlflowClient, model_name: str, version: str, status: str, reason: str | None = None
) -> None:
    tags = {STATUS_TAG: status}
    if reason:
        tags["status_reason"] = reason
    set_version_tags(client, model_name, version, tags)


def get_status(client: MlflowClient, model_name: str, version: str) -> str | None:
    return client.get_model_version(model_name, str(version)).tags.get(STATUS_TAG)


def promote(
    client: MlflowClient, model_name: str, version: str, alias: str = CHAMPION_ALIAS
) -> str | None:
    """Mark ``version`` approved and point ``alias`` to it.

    Returns the previous champion version (or None). The previous champion is recorded
    in the ``previous_champion`` tag of the new version, which rollback uses.
    """
    version = str(version)
    previous = get_version_by_alias(client, model_name, alias)
    # MLflow 3 may return the version as an int: normalise to str like everywhere else.
    previous_version = str(previous.version) if previous else None

    set_status(client, model_name, version, STATUS_APPROVED)
    if previous_version and previous_version != version:
        set_version_tags(client, model_name, version, {PREVIOUS_CHAMPION_TAG: previous_version})
    client.set_registered_model_alias(model_name, alias, version)
    return previous_version


def set_alias(client: MlflowClient, model_name: str, version: str, alias: str = CHAMPION_ALIAS):
    """Move an alias without touching tags (used by rollback / restore)."""
    client.set_registered_model_alias(model_name, alias, str(version))
