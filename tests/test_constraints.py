"""Tests for the constraints.physical check."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import constraints


def _const_predict(y):
    return lambda xx: y


def test_positivity_pass():
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(size=(200, 3)))
    result = constraints.physical(
        X_test=np.zeros((200, 1)),
        predict=_const_predict(y),
        metadata={"constraints": [{"type": "positivity"}]},
    )
    assert result.status is vc.Status.PASS


def test_positivity_fail():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 3))  # about half negative
    result = constraints.physical(
        X_test=np.zeros((200, 1)),
        predict=_const_predict(y),
        metadata={"constraints": [{"type": "positivity"}]},
    )
    assert result.status is vc.Status.FAIL
    assert result.metrics["max_violation_fraction"] > 0.01


def test_conservation_sum_pass_and_fail():
    n = 100
    good = np.full((n, 3), 1.0 / 3.0)  # rows sum to 1
    bad = np.ones((n, 3))  # rows sum to 3
    spec = {"constraints": [{"type": "sum", "value": 1.0, "axis": -1}]}
    r_good = constraints.physical(
        X_test=np.zeros((n, 1)), predict=_const_predict(good), metadata=spec
    )
    r_bad = constraints.physical(
        X_test=np.zeros((n, 1)), predict=_const_predict(bad), metadata=spec
    )
    assert r_good.status is vc.Status.PASS
    assert r_bad.status is vc.Status.FAIL


def test_monotonic_pass():
    rng = np.random.default_rng(2)
    y = np.cumsum(np.abs(rng.normal(size=(50, 10))), axis=-1)  # increasing rows
    result = constraints.physical(
        X_test=np.zeros((50, 1)),
        predict=_const_predict(y),
        metadata={"constraints": [{"type": "monotonic", "axis": -1}]},
    )
    assert result.status is vc.Status.PASS


def test_bounds_warn_on_small_fraction():
    y = np.full((1000, 1), 0.5)
    y[0, 0] = 2.0  # single value out of bounds -> fraction 0.001
    result = constraints.physical(
        X_test=np.zeros((1000, 1)),
        predict=_const_predict(y),
        metadata={"constraints": [{"type": "bounds", "low": 0.0, "high": 1.0}]},
    )
    assert result.status is vc.Status.WARN


def test_constraints_skip_without_specs():
    result = constraints.physical(
        X_test=np.zeros((10, 1)), predict=_const_predict(np.zeros((10, 1)))
    )
    assert result.status is vc.Status.SKIP


def test_unknown_constraint_type_skips():
    result = constraints.physical(
        X_test=np.zeros((10, 1)),
        predict=_const_predict(np.zeros((10, 1))),
        metadata={"constraints": [{"type": "not_a_real_constraint"}]},
    )
    assert result.status is vc.Status.SKIP
