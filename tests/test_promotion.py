"""Tests for the promotion and retraining rules."""

from __future__ import annotations

import pytest

from ml.promotion import PromotionConfig, decide, load_config, should_retrain


@pytest.fixture
def cfg() -> PromotionConfig:
    return PromotionConfig(
        metric="f1_spam",
        direction="maximize",
        quality_floor=0.85,
        min_improvement=0.01,
        retrain_when=("new_data_version", "champion_below_floor"),
    )


@pytest.fixture
def cfg_min() -> PromotionConfig:
    return PromotionConfig(
        metric="log_loss", direction="minimize", quality_floor=0.30, min_improvement=0.01
    )


# --- decide -----------------------------------------------------------------------------


def test_no_champion_above_floor_is_promoted(cfg):
    d = decide(0.90, None, cfg)
    assert d.promote and "no champion" in d.reason


def test_no_champion_exactly_at_floor_is_promoted(cfg):
    assert decide(0.85, None, cfg).promote


def test_no_champion_below_floor_is_rejected(cfg):
    d = decide(0.80, None, cfg)
    assert not d.promote and "floor" in d.reason


def test_better_by_002_is_promoted(cfg):
    assert decide(0.92, 0.90, cfg).promote


def test_better_by_exactly_min_improvement_is_promoted(cfg):
    assert decide(0.91, 0.90, cfg).promote


def test_better_by_0005_is_rejected(cfg):
    d = decide(0.905, 0.90, cfg)
    assert not d.promote and "improvement" in d.reason


def test_tie_is_rejected(cfg):
    assert not decide(0.90, 0.90, cfg).promote


def test_worse_candidate_is_rejected(cfg):
    assert not decide(0.88, 0.95, cfg).promote


def test_big_improvement_but_below_floor_is_rejected(cfg):
    d = decide(0.84, 0.50, cfg)
    assert not d.promote and "floor" in d.reason


def test_minimize_lower_is_better(cfg_min):
    assert decide(0.20, None, cfg_min).promote
    assert not decide(0.40, None, cfg_min).promote  # above floor = worse
    assert decide(0.20, 0.25, cfg_min).promote  # 0.05 lower
    assert not decide(0.245, 0.25, cfg_min).promote  # only 0.005 lower
    assert not decide(0.26, 0.25, cfg_min).promote  # worse


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        PromotionConfig(metric="f1", direction="up", quality_floor=0.5, min_improvement=0.0)


def test_repo_config_loads():
    cfg = load_config()
    assert cfg.metric == "f1_spam"
    assert cfg.direction == "maximize"
    assert cfg.quality_floor == 0.85
    assert cfg.min_improvement == 0.01


# --- should_retrain ---------------------------------------------------------------------


def test_retrain_without_champion(cfg):
    r = should_retrain("v1", None, None, cfg)
    assert r.retrain and r.reasons == ("no_champion",)


def test_retrain_on_new_data_version(cfg):
    r = should_retrain("v2", "v1", 0.95, cfg)
    assert r.retrain and r.reasons == ("new_data_version",)


def test_no_retrain_same_version_good_champion(cfg):
    r = should_retrain("v1", "v1", 0.95, cfg)
    assert not r.retrain and r.reason == "no reason to retrain"


def test_retrain_when_champion_below_floor(cfg):
    r = should_retrain("v1", "v1", 0.80, cfg)
    assert r.retrain and r.reasons == ("champion_below_floor",)


def test_both_reasons(cfg):
    r = should_retrain("v2", "v1", 0.80, cfg)
    assert r.reasons == ("new_data_version", "champion_below_floor")


def test_reasons_disabled_in_config():
    cfg = PromotionConfig("f1_spam", "maximize", 0.85, 0.01, retrain_when=())
    assert not should_retrain("v2", "v1", 0.10, cfg).retrain
