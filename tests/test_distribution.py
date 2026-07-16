"""Tests for the distribution checks."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import distribution


def test_coverage_fail_when_test_outside_training_domain():
    rng = np.random.default_rng(0)
    x_train = rng.uniform(0.0, 1.0, size=(500, 3))
    x_test = rng.uniform(2.0, 3.0, size=(100, 3))  # entirely outside train range
    result = distribution.coverage(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.FAIL
    assert result.metrics["frac_test_out_of_domain"] > 0.9


def test_coverage_pass_when_test_inside_training_domain():
    rng = np.random.default_rng(1)
    x_train = rng.uniform(0.0, 1.0, size=(1000, 3))
    x_test = rng.uniform(0.2, 0.8, size=(200, 3))  # strictly inside
    result = distribution.coverage(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.PASS
    assert result.metrics["frac_test_out_of_domain"] == 0.0


def test_coverage_warn_on_small_extrapolating_fraction():
    rng = np.random.default_rng(2)
    x_train = rng.uniform(0.0, 1.0, size=(1000, 3))
    x_test = rng.uniform(0.2, 0.8, size=(100, 3))
    x_test[:3] = 5.0  # 3% of points extrapolate
    result = distribution.coverage(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.WARN


def test_coverage_skips_without_test():
    rng = np.random.default_rng(3)
    result = distribution.coverage(X_train=rng.uniform(size=(50, 3)))
    assert result.status is vc.Status.SKIP


def test_coverage_attaches_figure_when_requested():
    rng = np.random.default_rng(4)
    x_train = rng.uniform(0.0, 1.0, size=(200, 3))
    x_test = rng.uniform(0.0, 1.0, size=(50, 3))
    result = distribution.coverage(
        X_train=x_train, X_test=x_test, metadata={"make_figures": True}
    )
    assert len(result.figures) == 1


def test_drift_fail_on_shifted_test_marginal():
    rng = np.random.default_rng(10)
    x_train = rng.normal(0.0, 1.0, size=(2000, 3))
    x_test = rng.normal(5.0, 1.0, size=(2000, 3))  # far-shifted marginals
    result = distribution.drift(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.FAIL
    assert result.metrics["max_ks_distance"] > 0.2


def test_drift_pass_on_matched_marginals():
    rng = np.random.default_rng(11)
    x_train = rng.normal(0.0, 1.0, size=(4000, 3))
    x_test = rng.normal(0.0, 1.0, size=(4000, 3))  # same distribution
    result = distribution.drift(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.PASS
    assert result.metrics["max_ks_distance"] < 0.1


def test_drift_warn_on_moderate_shift():
    rng = np.random.default_rng(12)
    x_train = rng.normal(0.0, 1.0, size=(4000, 2))
    x_test = rng.normal(0.3, 1.0, size=(4000, 2))  # ~0.12 KS, between thresholds
    result = distribution.drift(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.WARN


def test_drift_skips_without_heldout():
    rng = np.random.default_rng(13)
    result = distribution.drift(X_train=rng.normal(size=(100, 3)))
    assert result.status is vc.Status.SKIP
