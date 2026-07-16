"""Tests for ``Report.to_html``.

Covers the multi-check render, HTML escaping, the empty report, figure
embedding via a stub figure (so matplotlib is not needed), and the
file-writing path.
"""

from __future__ import annotations

import base64

import vibecheck as vc


class _StubFigure:
    """Minimal stand-in for a matplotlib Figure: just enough ``savefig``."""

    def __init__(self, payload: bytes = b"\x89PNG\r\n-stub-bytes"):
        self.payload = payload

    def savefig(self, buffer, *args, **kwargs):
        buffer.write(self.payload)


class _BrokenFigure:
    def savefig(self, buffer, *args, **kwargs):
        raise RuntimeError("cannot render")


def _report_with_two_checks() -> vc.Report:
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.pass",
            status=vc.Status.PASS,
            summary="everything looks fine",
            metrics={"max_error": 0.01},
        )
    )
    report.add(
        vc.CheckResult(
            name="demo.fail",
            status=vc.Status.FAIL,
            summary="a real problem",
            details="residuals exceed tolerance on the test split",
        )
    )
    return report


def test_html_renders_checks_and_overall_status():
    html = _report_with_two_checks().to_html()
    assert "<!doctype html>" in html
    assert "demo.pass" in html
    assert "demo.fail" in html
    assert "max_error" in html
    assert "residuals exceed tolerance" in html
    # Worst status across PASS and FAIL is FAIL.
    assert "Overall:" in html
    assert "FAIL" in html


def test_html_escapes_text():
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.xss",
            status=vc.Status.WARN,
            summary="<script>alert(1)</script>",
        )
    )
    html = report.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_empty_report_html():
    html = vc.Report().to_html()
    assert "No checks were run." in html
    # An empty report summarizes as SKIP.
    assert "SKIP" in html


def test_html_embeds_figure():
    payload = b"\x89PNG\r\n-stub-bytes"
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.figure",
            status=vc.Status.PASS,
            summary="with a figure",
            figures=[_StubFigure(payload)],
        )
    )
    html = report.to_html()
    expected = base64.b64encode(payload).decode("ascii")
    assert "data:image/png;base64," in html
    assert expected in html


def test_html_notes_unrenderable_figures():
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.badfigure",
            status=vc.Status.PASS,
            summary="figure objects that cannot be drawn",
            # One has no savefig at all, one raises inside savefig.
            figures=[object(), _BrokenFigure()],
        )
    )
    html = report.to_html()
    assert html.count("(figure could not be rendered)") == 2
    assert "data:image/png;base64," not in html


def test_html_writes_file(tmp_path):
    out = tmp_path / "report.html"
    text = _report_with_two_checks().to_html(path=str(out))
    assert out.read_text(encoding="utf-8") == text
    assert "demo.pass" in out.read_text(encoding="utf-8")
