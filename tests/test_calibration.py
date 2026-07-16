"""Tests for the calibration.coverage check."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import calibration


def _standard_normal_targets(n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, size=(n, 1))
    x = np.zeros((n, 1))
    return x, y


def test_calibration_pass_when_std_matches_noise():
    x, y = _standard_normal_targets(seed=0)
    result = calibration.coverage(
        X_test=x, y_test=y, predict=lambda xx: np.zeros((xx.shape[0], 1)),
        metadata={"predicted_std": np.ones_like(y)},
    )
    assert result.status is vc.Status.PASS


def test_calibration_fail_when_overconfident():
    x, y = _standard_normal_targets(seed=1)
    result = calibration.coverage(
        X_test=x, y_test=y, predict=lambda xx: np.zeros((xx.shape[0], 1)),
        metadata={"predicted_std": 0.3},  # far too small
    )
    assert result.status is vc.Status.FAIL
    assert result.metrics["empirical_coverage_1sigma"] < result.metrics[
        "nominal_coverage_1sigma"
    ]


def test_calibration_warn_when_slightly_overconfident():
    x, y = _standard_normal_targets(seed=2)
    result = calibration.coverage(
        X_test=x, y_test=y, predict=lambda xx: np.zeros((xx.shape[0], 1)),
        metadata={"predicted_std": 0.8},
    )
    assert result.status is vc.Status.WARN


def test_calibration_accepts_mean_std_tuple():
    x, y = _standard_normal_targets(seed=3)
    result = calibration.coverage(
        X_test=x,
        y_test=y,
        predict=lambda xx: (np.zeros((xx.shape[0], 1)), np.ones((xx.shape[0], 1))),
    )
    assert result.status is vc.Status.PASS


def test_calibration_skips_without_uncertainty():
    x, y = _standard_normal_targets(seed=4)
    result = calibration.coverage(
        X_test=x, y_test=y, predict=lambda xx: np.zeros((xx.shape[0], 1))
    )
    assert result.status is vc.Status.SKIP


def test_calibration_attaches_figure_when_requested():
    x, y = _standard_normal_targets(seed=5)
    result = calibration.coverage(
        X_test=x, y_test=y, predict=lambda xx: np.zeros((xx.shape[0], 1)),
        metadata={"predicted_std": np.ones_like(y), "make_figures": True},
    )
    assert len(result.figures) == 1
