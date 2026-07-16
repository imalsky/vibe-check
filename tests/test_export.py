"""Tests for the export.roundtrip check."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import export


def _base_predict(w):
    return lambda xx: np.asarray(xx) @ w


def test_export_pass_when_paths_agree():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))
    w = rng.normal(size=(4, 2))
    predict = _base_predict(w)
    result = export.roundtrip(
        X_test=x, predict=predict, metadata={"exported_predict": predict}
    )
    assert result.status is vc.Status.PASS


def test_export_pass_within_tolerance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 4))
    w = rng.normal(size=(4, 2))
    predict = _base_predict(w)

    def exported(xx):
        return predict(xx) + 1e-8  # tiny, within atol

    result = export.roundtrip(
        X_test=x, predict=predict, metadata={"exported_predict": exported}
    )
    assert result.status is vc.Status.PASS


def test_export_fail_when_paths_diverge():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 4))
    w = rng.normal(size=(4, 2))
    predict = _base_predict(w)

    def exported(xx):
        return predict(xx) + 0.1  # large, everywhere

    result = export.roundtrip(
        X_test=x, predict=predict, metadata={"exported_predict": exported}
    )
    assert result.status is vc.Status.FAIL
    assert result.metrics["fraction_mismatched"] > 0.01


def test_export_warn_on_small_fraction():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5))
    w = rng.normal(size=(5, 5))  # 200 x 5 = 1000 outputs
    predict = _base_predict(w)

    def exported(xx):
        out = predict(xx).copy()
        out[0, 0] += 1.0  # 1 of 1000 outputs differs -> 0.1%
        return out

    result = export.roundtrip(
        X_test=x, predict=predict, metadata={"exported_predict": exported}
    )
    assert result.status is vc.Status.WARN


def test_export_fail_on_shape_mismatch():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 4))
    w = rng.normal(size=(4, 2))
    predict = _base_predict(w)
    result = export.roundtrip(
        X_test=x, predict=predict, metadata={"exported_predict": lambda xx: xx}
    )
    assert result.status is vc.Status.FAIL


def test_export_skips_without_exported_model():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(10, 4))
    result = export.roundtrip(X_test=x, predict=lambda xx: xx)
    assert result.status is vc.Status.SKIP
