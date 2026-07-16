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
    assert vc.Status.PASS.rank > vc.Status.SKIP.rank


def _report_with(*statuses: vc.Status) -> vc.Report:
    report = vc.Report()
    for i, status in enumerate(statuses):
        report.add(vc.CheckResult(name=f"demo.{i}", status=status, summary="x"))
    return report


def test_summary_pass_beats_skip_regardless_of_order():
    # A report that passed something must not headline as SKIP; this was
    # order-dependent when PASS and SKIP shared a rank.
    assert _report_with(vc.Status.SKIP, vc.Status.PASS).summary() is vc.Status.PASS
    assert _report_with(vc.Status.PASS, vc.Status.SKIP).summary() is vc.Status.PASS
    assert _report_with(vc.Status.SKIP, vc.Status.SKIP).summary() is vc.Status.SKIP


def test_report_str_lists_checks():
    text = str(_report_with(vc.Status.PASS, vc.Status.WARN))
    assert text.startswith("vibe-check: WARN")
    assert "demo.0: PASS" in text
    assert "demo.1: WARN" in text


def test_unknown_metadata_key_warns():
    import warnings

    import pytest

    def predict(x):
        return np.asarray(x)

    with pytest.warns(UserWarning, match="normalisation.*did you mean 'normalization'"):
        vc.check(predict=predict, X_test=np.zeros((4, 2)), metadata={"normalisation": {}})
    # Known keys stay silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vc.check(predict=predict, X_test=np.zeros((4, 2)), metadata={"make_figures": False})


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
