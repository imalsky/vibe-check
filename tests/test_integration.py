"""End-to-end tests: the orchestrator runs every registered check and renders.

These exercise ``vibecheck.check`` over a small real surrogate, covering the
full contract: the registry, SKIP vs PASS vs FAIL, exception wrapping, and both
report renderers including embedded figures.
"""

from __future__ import annotations

import numpy as np

import vibecheck as vc
from vibecheck import core


def _fitted_surrogate(seed: int = 0, noise: float = 0.1):
    rng = np.random.default_rng(seed)
    d_in, d_out = 4, 2
    weight = rng.normal(size=(d_in, d_out))

    def sample(n: int):
        x = rng.normal(size=(n, d_in))
        y = x @ weight + rng.normal(0.0, noise, size=(n, d_out))
        return x, y

    x_train, y_train = sample(400)
    x_val, y_val = sample(100)
    x_test, y_test = sample(200)
    fit, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

    def predict(x):
        return np.asarray(x) @ fit

    resid = predict(x_val) - y_val
    sigma = float(resid.std())
    data = dict(
        X_train=x_train, y_train=y_train,
        X_val=x_val, y_val=y_val,
        X_test=x_test, y_test=y_test,
    )
    return predict, data, sigma


def _clean_metadata(predict, data, sigma):
    return {
        "normalization": {
            "mean": data["X_train"].mean(axis=0),
            "std": data["X_train"].std(axis=0),
        },
        "predicted_std": np.full_like(data["y_test"], sigma),
        "constraints": [{"type": "bounds", "low": -1e6, "high": 1e6}],
        "exported_predict": predict,
        "speed": {"min_throughput": 1.0},
    }


def test_check_runs_every_registered_check():
    predict, data, sigma = _fitted_surrogate()
    report = vc.check(predict, metadata=_clean_metadata(predict, data, sigma), **data)
    assert len(report.results) == len(core._REGISTERED_CHECKS) == 10
    names = {r.name for r in report.results}
    assert "leakage.normalization" in names
    assert "speed.inference" in names


def test_clean_surrogate_has_no_failures():
    predict, data, sigma = _fitted_surrogate()
    report = vc.check(predict, metadata=_clean_metadata(predict, data, sigma), **data)
    statuses = {r.name: r.status for r in report.results}
    failed = {n: s for n, s in statuses.items() if s is vc.Status.FAIL}
    assert not failed, f"unexpected failures: {failed}"
    # error.field does not apply to a small tabular output.
    assert statuses["error.field"] is vc.Status.SKIP


def test_leaked_normalization_makes_report_fail():
    predict, data, sigma = _fitted_surrogate()
    meta = _clean_metadata(predict, data, sigma)
    x_all = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
    meta["normalization"] = {"mean": x_all.mean(axis=0), "std": x_all.std(axis=0)}
    report = vc.check(predict, metadata=meta, **data)
    assert report.summary() is vc.Status.FAIL
    norm = next(r for r in report.results if r.name == "leakage.normalization")
    assert norm.status is vc.Status.FAIL


def test_report_renders_markdown_and_html_with_figures():
    predict, data, sigma = _fitted_surrogate()
    meta = _clean_metadata(predict, data, sigma)
    meta["make_figures"] = True
    report = vc.check(predict, metadata=meta, **data)

    md = report.to_markdown()
    assert "vibe-check report" in md
    assert "leakage.normalization" in md

    html = report.to_html()
    assert "<!doctype html>" in html
    # At least one check attached a figure, embedded as base64 PNG.
    assert "data:image/png;base64," in html


def test_orchestrator_wraps_check_exceptions_as_error():
    def broken(**context):
        raise ValueError("boom")

    broken.check_name = "test.broken"
    core._REGISTERED_CHECKS.append(broken)
    try:
        report = vc.check(lambda x: x, X_test=np.zeros((3, 2)))
    finally:
        core._REGISTERED_CHECKS.remove(broken)
    errored = [r for r in report.results if r.name == "test.broken"]
    assert errored and errored[0].status is vc.Status.ERROR
