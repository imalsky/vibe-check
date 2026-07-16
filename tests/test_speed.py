"""Tests for the speed.inference check.

The verdicts are made deterministic by using trivially-met or impossible
budgets, so they do not depend on how fast the test machine is.
"""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import speed


def _identity(xx):
    return np.asarray(xx)


def test_speed_pass_with_easy_budget():
    x = np.zeros((100, 4))
    result = speed.inference(
        X_test=x, predict=_identity, metadata={"speed": {"min_throughput": 1.0}}
    )
    assert result.status is vc.Status.PASS
    assert result.metrics["throughput_samples_per_s"] > 1.0


def test_speed_fail_with_impossible_throughput():
    x = np.zeros((100, 4))
    result = speed.inference(
        X_test=x, predict=_identity, metadata={"speed": {"min_throughput": 1e15}}
    )
    assert result.status is vc.Status.FAIL


def test_speed_fail_with_zero_latency_budget():
    x = np.zeros((100, 4))
    result = speed.inference(
        X_test=x, predict=_identity, metadata={"speed": {"max_latency_s": 0.0}}
    )
    assert result.status is vc.Status.FAIL


def test_speed_echoes_budget_thresholds_in_metrics():
    x = np.zeros((50, 4))
    result = speed.inference(
        X_test=x,
        predict=_identity,
        metadata={"speed": {"min_throughput": 1.0, "max_latency_s": 10.0}},
    )
    assert result.metrics["min_throughput"] == 1.0
    assert result.metrics["max_latency_s"] == 10.0


def test_speed_skips_without_budget_but_reports_numbers():
    x = np.zeros((100, 4))
    result = speed.inference(X_test=x, predict=_identity)
    assert result.status is vc.Status.SKIP
    assert result.metrics["throughput_samples_per_s"] > 0.0
    assert "min_throughput" not in result.metrics
    assert "max_latency_s" not in result.metrics


def test_speed_skips_without_inputs():
    result = speed.inference(predict=_identity)
    assert result.status is vc.Status.SKIP
