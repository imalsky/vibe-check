"""Tests for the error checks."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import error


def _linear_data(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    w = rng.normal(size=(4, 2))
    y = x @ w
    return x, y, w


def test_pointwise_pass_for_accurate_surrogate():
    x, y, w = _linear_data()

    def predict(xx):
        return xx @ w  # exact

    result = error.pointwise(X_test=x, y_test=y, y_train=y, predict=predict)
    assert result.status is vc.Status.PASS
    assert result.metrics["skill_vs_mean_baseline"] > 0.99


def test_pointwise_fail_when_worse_than_baseline():
    x, y, _ = _linear_data(seed=1)
    y = y + 10.0  # non-zero mean, so the mean baseline is informative

    def predict(xx):
        return np.zeros((xx.shape[0], 2))  # ignores input, worse than the mean

    result = error.pointwise(X_test=x, y_test=y, y_train=y, predict=predict)
    assert result.status is vc.Status.FAIL
    assert result.metrics["skill_vs_mean_baseline"] <= 0


def test_pointwise_fail_on_nan_predictions():
    x, y, w = _linear_data(seed=2)

    def predict(xx):
        out = xx @ w
        out[0, 0] = np.nan
        return out

    result = error.pointwise(X_test=x, y_test=y, predict=predict)
    assert result.status is vc.Status.FAIL


def test_pointwise_warn_on_moderate_skill():
    rng = np.random.default_rng(3)
    n = 5000
    y = rng.normal(size=(n, 1))
    x = y.copy()
    noise = rng.normal(scale=np.sqrt(0.7), size=(n, 1))

    def predict(xx):
        return xx + noise  # r2 ~ 0.3, between 0 and warn_skill

    result = error.pointwise(X_test=x, y_test=y, predict=predict)
    assert result.status is vc.Status.WARN


def test_pointwise_fail_when_rmse_budget_exceeded():
    x, y, w = _linear_data(seed=4)

    def predict(xx):
        return xx @ w + 0.5  # small but non-zero error, RMSE = 0.5

    result = error.pointwise(
        X_test=x, y_test=y, predict=predict, metadata={"error": {"max_rmse": 1e-3}}
    )
    assert result.status is vc.Status.FAIL
    assert result.metrics["rmse"] > 1e-3


def test_pointwise_skips_without_targets():
    x, _, w = _linear_data(seed=5)
    result = error.pointwise(X_test=x, predict=lambda xx: xx @ w)
    assert result.status is vc.Status.SKIP


def test_pointwise_attaches_figures_when_requested():
    x, y, w = _linear_data(seed=6)
    result = error.pointwise(
        X_test=x, y_test=y, predict=lambda xx: xx @ w, metadata={"make_figures": True}
    )
    assert len(result.figures) == 2
