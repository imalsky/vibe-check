"""Smoke tests: the package imports and the core contract behaves.

These stay green from the first commit. Real per-check tests are added
alongside each diagnostic as it is implemented.
"""

from __future__ import annotations

import numpy as np

import vibecheck as vc


def test_imports_and_version():
    assert isinstance(vc.__version__, str)
    assert hasattr(vc, "check")


def test_empty_report_is_skip():
    report = vc.Report()
    assert report.summary() is vc.Status.SKIP


def test_status_ranking():
    assert vc.Status.FAIL.rank > vc.Status.WARN.rank > vc.Status.PASS.rank
    assert vc.Status.ERROR.rank > vc.Status.FAIL.rank


def test_check_runs_registered_checks():
    def predict(x):
        return np.asarray(x)

    report = vc.check(predict=predict, X_test=np.zeros((4, 2)), y_test=np.zeros((4, 2)))
    assert isinstance(report, vc.Report)
    # At least one check is registered, and none should raise (ERROR) or
    # silently pass when their required inputs are missing.
    assert report.results
    assert all(r.status is not vc.Status.ERROR for r in report.results)


def test_report_markdown_roundtrip():
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.example",
            status=vc.Status.PASS,
            summary="everything looks fine",
            metrics={"max_error": 0.01},
        )
    )
    text = report.to_markdown()
    assert "demo.example" in text
    assert "max_error" in text
