"""Pure promotion and retraining decision functions (no I/O except config loading)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("ml/config/promotion.yaml")
# Tolerance so that an improvement of exactly min_improvement is not lost to float rounding.
EPS = 1e-9
RETRAIN_REASONS = {"new_data_version", "champion_below_floor"}


@dataclass(frozen=True)
class PromotionConfig:
    metric: str
    direction: str  # "maximize" or "minimize"
    quality_floor: float
    min_improvement: float
    retrain_when: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"direction must be maximize or minimize, got {self.direction!r}")
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be >= 0")
        unknown = set(self.retrain_when) - RETRAIN_REASONS
        if unknown:
            raise ValueError(f"unknown retrain_when entries: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, data: dict) -> PromotionConfig:
        return cls(
            metric=data["metric"],
            direction=data["direction"],
            quality_floor=float(data["quality_floor"]),
            min_improvement=float(data["min_improvement"]),
            retrain_when=tuple(data.get("retrain_when") or ()),
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> PromotionConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return PromotionConfig.from_dict(yaml.safe_load(fh))


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return ", ".join(self.reasons) if self.reasons else "no reason to retrain"


def passes_floor(score: float, cfg: PromotionConfig) -> bool:
    if cfg.direction == "maximize":
        return score >= cfg.quality_floor - EPS
    return score <= cfg.quality_floor + EPS


def improvement(candidate_score: float, champion_score: float, cfg: PromotionConfig) -> float:
    """Signed gain of the candidate over the champion (positive = better)."""
    if cfg.direction == "maximize":
        return candidate_score - champion_score
    return champion_score - candidate_score


def decide(candidate_score: float, champion_score: float | None, cfg: PromotionConfig) -> Decision:
    """Promotion rule.

    - No champion: promote if the candidate reaches the quality floor.
    - With a champion: the candidate must reach the floor AND beat the champion
      by at least ``min_improvement`` (in the direction of the metric).
    """
    m = cfg.metric
    if not passes_floor(candidate_score, cfg):
        return Decision(
            False,
            f"rejected: candidate {m}={candidate_score:.4f} does not reach the quality "
            f"floor {cfg.quality_floor} ({cfg.direction})",
        )
    if champion_score is None:
        return Decision(
            True,
            f"promoted: no champion and candidate {m}={candidate_score:.4f} reaches the floor "
            f"{cfg.quality_floor}",
        )
    gain = improvement(candidate_score, champion_score, cfg)
    if gain + EPS < cfg.min_improvement:
        return Decision(
            False,
            f"rejected: candidate {m}={candidate_score:.4f} vs champion {champion_score:.4f}, "
            f"improvement {gain:+.4f} < required {cfg.min_improvement}",
        )
    return Decision(
        True,
        f"promoted: candidate {m}={candidate_score:.4f} vs champion {champion_score:.4f}, "
        f"improvement {gain:+.4f} >= {cfg.min_improvement}",
    )


def should_retrain(
    requested_version: str,
    champion_data_version: str | None,
    champion_score: float | None,
    cfg: PromotionConfig,
) -> RetrainDecision:
    """Retraining rule, driven by ``cfg.retrain_when``.

    Without a champion we always train (bootstrap). A champion with an unknown score
    is not considered below the floor.
    """
    if champion_data_version is None and champion_score is None:
        return RetrainDecision(True, ("no_champion",))

    reasons = []
    if "new_data_version" in cfg.retrain_when and requested_version != champion_data_version:
        reasons.append("new_data_version")
    if (
        "champion_below_floor" in cfg.retrain_when
        and champion_score is not None
        and not passes_floor(champion_score, cfg)
    ):
        reasons.append("champion_below_floor")
    return RetrainDecision(bool(reasons), tuple(reasons))
