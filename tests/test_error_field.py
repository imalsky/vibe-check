"""Tests for the error.field check."""

from __future__ import annotations

import numpy as np
import pytest

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
    pytest.importorskip("matplotlib")
    y = _fields(5)
    x = np.zeros((y.shape[0], 2))
    result = error.field(
        X_test=x, y_test=y, y_train=y, predict=lambda xx: y + 1e-4,
        metadata={"make_figures": True},
    )
    assert len(result.figures) == 1


def test_field_figure_for_1d_field():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(6)
    y = rng.normal(size=(30, 32))  # 1-D spatial field
    x = np.zeros((30, 2))
    result = error.field(
        X_test=x, y_test=y, y_train=y, predict=lambda xx: y + 1e-4,
        metadata={"make_figures": True},
    )
    assert len(result.figures) == 1


def test_field_fail_on_nonfinite_predictions_with_figures():
    pytest.importorskip("matplotlib")
    y = _fields(7)
    x = np.zeros((y.shape[0], 2))

    def predict(xx):
        out = y.copy()
        out[0, 0, 0] = np.nan
        return out

    result = error.field(
        X_test=x, y_test=y, y_train=y, predict=predict,
        metadata={"make_figures": True},
    )
    assert result.status is vc.Status.FAIL
    assert result.figures == []


def test_field_fail_on_transposed_predictions():
    y = _fields(8)  # (40, 8, 8)
    x = np.zeros((y.shape[0], 2))

    def predict(xx):
        return np.transpose(y, (1, 2, 0))  # (8, 8, 40): same size, wrong layout

    result = error.field(X_test=x, y_test=y, y_train=y, predict=predict)
    assert result.status is vc.Status.FAIL
    assert "shape" in result.summary
    assert "skill" not in result.summary


def test_field_falls_back_when_y_train_shape_mismatches():
    y = _fields(9)
    x = np.zeros((y.shape[0], 2))
    y_train = np.zeros((30, 5))  # cannot be pooled into (8, 8) fields
    result = error.field(
        X_test=x, y_test=y, y_train=y_train, predict=lambda xx: y + 1e-4
    )
    assert result.status is vc.Status.PASS
    assert "derived from y_test instead" in result.details


def test_field_worst_sample_matches_reported_percent_error():
    # Sample 0 has the larger absolute error, sample 1 the larger percent error.
    y = np.stack([np.full((8, 8), 100.0), np.full((8, 8), 1.0)])
    y_hat = y.copy()
    y_hat[0] += 5.0  # MAE 5, percent error 5
    y_hat[1] += 0.5  # MAE 0.5, percent error 50
    x = np.zeros((2, 2))
    result = error.field(X_test=x, y_test=y, y_train=y, predict=lambda xx: y_hat)
    assert result.metrics["worst_sample_index"] == 1.0
    assert abs(result.metrics["max_abs_percent_error"] - 50.0) < 1e-6


def test_field_echoes_thresholds_in_metrics():
    y = _fields(10)
    x = np.zeros((y.shape[0], 2))
    result = error.field(X_test=x, y_test=y, y_train=y, predict=lambda xx: y + 1e-4)
    assert result.metrics["warn_pct"] == 5.0
    assert result.metrics["fail_pct"] == 20.0
