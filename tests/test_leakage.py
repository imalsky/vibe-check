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


def test_normalization_handles_zero_mean_features():
    # Regression test: centered features have a near-zero mean, and a mean gap
    # normalized by the mean itself would blow up for correct train-only stats.
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(200, 3))
    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_test = rng.normal(size=(100, 3))
    x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0) * 2.0

    # Train-only statistics recomputed in float32 arithmetic: absolutely tiny
    # discrepancies, huge relative to the near-zero mean. Must PASS.
    norm_train = {
        "mean": x_train.astype(np.float32).mean(axis=0),
        "std": x_train.astype(np.float32).std(axis=0),
    }
    result = leakage.normalization(
        X_train=x_train, X_test=x_test, metadata={"normalization": norm_train}
    )
    assert result.status is vc.Status.PASS

    # Full-data statistics must still be caught as leakage.
    x_all = np.concatenate([x_train, x_test], axis=0)
    norm_full = {"mean": x_all.mean(axis=0), "std": x_all.std(axis=0)}
    result_full = leakage.normalization(
        X_train=x_train, X_test=x_test, metadata={"normalization": norm_full}
    )
    assert result_full.status is vc.Status.FAIL


def test_normalization_skips_without_stats():
    x_train, x_test = _split()
    result = leakage.normalization(X_train=x_train, X_test=x_test, metadata={})
    assert result.status is vc.Status.SKIP


def test_normalization_skips_without_heldout_split():
    x_train, _ = _split()
    norm = {"mean": x_train.mean(axis=0), "std": x_train.std(axis=0)}
    result = leakage.normalization(X_train=x_train, metadata={"normalization": norm})
    assert result.status is vc.Status.SKIP


def test_split_overlap_fail_on_exact_duplicates():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(120, 4))
    x_test = rng.normal(size=(60, 4))
    x_test[:5] = x_train[:5]  # five rows leaked verbatim into the test split
    result = leakage.split_overlap(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.FAIL
    assert result.metrics["exact_duplicate_rows"] >= 5


def test_split_overlap_pass_on_disjoint_splits():
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(120, 4))
    x_test = rng.normal(size=(60, 4)) + 100.0  # far apart, no overlap
    result = leakage.split_overlap(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.PASS
    assert result.metrics["exact_duplicate_rows"] == 0
    assert result.metrics["near_duplicate_rows"] == 0


def test_split_overlap_warn_on_near_duplicates():
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(120, 4))
    x_test = rng.normal(size=(60, 4)) + 100.0
    x_test[:5] = x_train[:5] + 1e-5  # near-duplicates, not exact
    result = leakage.split_overlap(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.WARN
    assert result.metrics["near_duplicate_rows"] >= 5
    assert result.metrics["exact_duplicate_rows"] == 0


def test_split_overlap_skips_with_one_split():
    rng = np.random.default_rng(4)
    result = leakage.split_overlap(X_train=rng.normal(size=(30, 4)))
    assert result.status is vc.Status.SKIP
