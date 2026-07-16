"""Tests for the error.field check."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import error


def _fields(seed: int = 0, n: int = 40, h: int = 8, w: int = 8):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, h, w))


def test_field_pass_for_accurate_surrogate():
    y = _fields(0)
    x = np.zeros((y.shape[0], 2))
    result = error.field(X_test=x, y_test=y, y_train=y, predict=lambda xx: y + 1e-4)
    assert result.status is vc.Status.PASS


def test_field_fail_without_skill():
    y = _fields(1)
    x = np.zeros((y.shape[0], 2))
    result = error.field(
        X_test=x, y_test=y, y_train=y, predict=lambda xx: np.zeros_like(y)
    )
    assert result.status is vc.Status.FAIL


def test_field_fail_on_large_percent_error():
    y = _fields(2)
    x = np.zeros((y.shape[0], 2))
    result = error.field(X_test=x, y_test=y, y_train=y, predict=lambda xx: y * 1.5)
    assert result.status is vc.Status.FAIL
    assert result.metrics["mean_abs_percent_error"] > 20.0


def test_field_warn_on_moderate_percent_error():
    y = _fields(3)
    x = np.zeros((y.shape[0], 2))
    result = error.field(X_test=x, y_test=y, y_train=y, predict=lambda xx: y * 1.10)
    assert result.status is vc.Status.WARN


def test_field_skips_small_tabular_output():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(40, 3))  # below the field-size threshold
    x = np.zeros((40, 2))
    result = error.field(X_test=x, y_test=y, predict=lambda xx: y)
    assert result.status is vc.Status.SKIP


def test_field_skips_without_targets():
    result = error.field(X_test=np.zeros((4, 2)), predict=lambda xx: xx)
    assert result.status is vc.Status.SKIP


def test_field_figure_for_2d_field():
    y = _fields(5)
    x = np.zeros((y.shape[0], 2))
    result = error.field(
        X_test=x, y_test=y, y_train=y, predict=lambda xx: y + 1e-4,
        metadata={"make_figures": True},
    )
    assert len(result.figures) == 1


def test_field_figure_for_1d_field():
    rng = np.random.default_rng(6)
    y = rng.normal(size=(30, 32))  # 1-D spatial field
    x = np.zeros((30, 2))
    result = error.field(
        X_test=x, y_test=y, y_train=y, predict=lambda xx: y + 1e-4,
        metadata={"make_figures": True},
    )
    assert len(result.figures) == 1
