"""Tests for the distribution checks."""

from __future__ import annotations

import numpy as np
import pytest

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
    x_test[:6] = 5.0  # 6% of points extrapolate: above warn, below fail
    result = distribution.coverage(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.WARN
    assert "above warn threshold" in result.summary


def test_coverage_skips_without_test():
    rng = np.random.default_rng(3)
    result = distribution.coverage(X_train=rng.uniform(size=(50, 3)))
    assert result.status is vc.Status.SKIP


def test_coverage_attaches_figure_when_requested():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(4)
    x_train = rng.uniform(0.0, 1.0, size=(200, 3))
    x_test = rng.uniform(0.0, 1.0, size=(50, 3))
    result = distribution.coverage(
        X_train=x_train, X_test=x_test, metadata={"make_figures": True}
    )
    assert len(result.figures) == 1


def test_coverage_all_pass_figure_has_zero_ylim_bottom():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(5)
    x_train = rng.uniform(0.0, 1.0, size=(500, 3))
    x_test = rng.uniform(0.2, 0.8, size=(100, 3))  # strictly inside envelope
    result = distribution.coverage(
        X_train=x_train, X_test=x_test, metadata={"make_figures": True}
    )
    assert result.metrics["frac_test_out_of_domain"] == 0.0
    assert len(result.figures) == 1
    ax = result.figures[0].axes[0]
    assert ax.get_ylim()[0] == 0


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


def test_drift_attaches_two_figures_when_requested():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(14)
    x_train = rng.normal(size=(300, 3))
    x_test = rng.normal(size=(150, 3))
    result = distribution.drift(
        X_train=x_train, X_test=x_test, metadata={"make_figures": True}
    )
    assert len(result.figures) == 2


def test_defaults_do_not_warn_on_iid_data_across_seeds():
    # Regression test for the fixed defaults that warned on healthy iid data
    # in most seeds. The adaptive defaults must PASS all of these.
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x_train = rng.normal(size=(400, 4))
        x_test = rng.normal(size=(200, 4))
        cov = distribution.coverage(X_train=x_train, X_test=x_test)
        dr = distribution.drift(X_train=x_train, X_test=x_test)
        assert cov.status is vc.Status.PASS, f"coverage not PASS at seed {seed}"
        assert dr.status is vc.Status.PASS, f"drift not PASS at seed {seed}"


def test_drift_still_fails_on_genuine_shift_at_small_sample_size():
    rng = np.random.default_rng(42)
    x_train = rng.normal(0.0, 1.0, size=(400, 4))
    x_test = rng.normal(3.0, 1.0, size=(200, 4))  # shifted by 3 std
    result = distribution.drift(X_train=x_train, X_test=x_test)
    assert result.status is vc.Status.FAIL


def test_explicit_metadata_thresholds_are_used_verbatim():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(50, 2))
    x_test = rng.normal(size=(30, 2))
    dr = distribution.drift(
        X_train=x_train,
        X_test=x_test,
        metadata={"drift": {"warn_ks": 0.01, "fail_ks": 0.02}},
    )
    assert dr.metrics["warn_ks"] == 0.01
    assert dr.metrics["fail_ks"] == 0.02
    assert dr.status is vc.Status.FAIL  # tiny iid samples exceed a 0.02 KS bar
    cov = distribution.coverage(
        X_train=x_train,
        X_test=x_test,
        metadata={"coverage": {"warn_frac": 0.5, "fail_frac": 0.9}},
    )
    assert cov.metrics["warn_frac"] == 0.5
    assert cov.metrics["fail_frac"] == 0.9
