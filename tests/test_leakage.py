"""Tests for the leakage checks."""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck.checks import leakage


def _split(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Train and test are drawn from different means, so the train-only and
    # full-data statistics genuinely differ and leakage is detectable.
    x_train = rng.normal(0.0, 1.0, size=(200, 3))
    x_test = rng.normal(5.0, 2.0, size=(200, 3))
    return x_train, x_test


def test_normalization_pass_when_fit_on_train_only():
    x_train, x_test = _split()
    norm = {"mean": x_train.mean(axis=0), "std": x_train.std(axis=0)}
    result = leakage.normalization(
        X_train=x_train, X_test=x_test, metadata={"normalization": norm}
    )
    assert result.status is vc.Status.PASS


def test_normalization_fail_when_fit_on_full_data():
    x_train, x_test = _split()
    x_all = np.concatenate([x_train, x_test], axis=0)
    norm = {"mean": x_all.mean(axis=0), "std": x_all.std(axis=0)}
    result = leakage.normalization(
        X_train=x_train, X_test=x_test, metadata={"normalization": norm}
    )
    assert result.status is vc.Status.FAIL
    assert result.metrics["rel_distance_to_full_stats"] < result.metrics[
        "rel_distance_to_train_stats"
    ]


def test_normalization_skips_without_stats():
    x_train, x_test = _split()
    result = leakage.normalization(X_train=x_train, X_test=x_test, metadata={})
    assert result.status is vc.Status.SKIP


def test_normalization_skips_without_heldout_split():
    x_train, _ = _split()
    norm = {"mean": x_train.mean(axis=0), "std": x_train.std(axis=0)}
    result = leakage.normalization(X_train=x_train, metadata={"normalization": norm})
    assert result.status is vc.Status.SKIP
