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


def test_html_overview_table_links_sections_worst_first():
    html = _report_with_two_checks().to_html()
    assert '<table class="overview">' in html
    # The failing check is listed before the passing one in the overview.
    assert html.index('href="#check-demo-fail"') < html.index('href="#check-demo-pass"')
    assert 'id="check-demo-fail"' in html
    assert 'id="check-demo-pass"' in html


def test_html_details_preserve_line_structure():
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.details",
            status=vc.Status.PASS,
            summary="multi-line details",
            details="- first line\n- second line",
        )
    )
    html = report.to_html()
    assert 'class="details"' in html
    assert "- first line\n- second line" in html


def test_custom_title_appears_in_both_renderers():
    report = _report_with_two_checks()
    html = report.to_html(title="vibe-check report: demo surrogate")
    md = report.to_markdown(title="vibe-check report: demo surrogate")
    assert "<title>vibe-check report: demo surrogate</title>" in html
    assert md.startswith("# vibe-check report: demo surrogate")


def test_metric_display_formatting():
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.metrics",
            status=vc.Status.PASS,
            summary="formatted metrics",
            metrics={"count": 1401.0, "index": 2.0, "rmse": 0.0014506988575394902},
        )
    )
    for text in (report.to_markdown(), report.to_html()):
        assert "1401.0" not in text
        assert "0.0014506988575394902" not in text
    md = report.to_markdown()
    assert "`count`: 1401" in md
    assert "`index`: 2" in md
    assert "`rmse`: 0.001451" in md


def test_markdown_summary_table_and_figure_note():
    report = _report_with_two_checks()
    report.results[0].figures = [_StubFigure()]
    md = report.to_markdown()
    assert "| check | status | summary |" in md
    # Worst first: the FAIL row precedes the PASS row in the table.
    assert md.index("| demo.fail | FAIL |") < md.index("| demo.pass | PASS |")
    assert "(1 figure attached; see the HTML report)" in md


def test_markdown_empty_report_notes_no_checks():
    md = vc.Report().to_markdown()
    assert "No checks were run." in md


def test_markdown_collapses_newlines_in_name_and_summary():
    report = vc.Report()
    report.add(
        vc.CheckResult(
            name="demo.inject",
            status=vc.Status.PASS,
            summary="line one\n## fake heading",
        )
    )
    md = report.to_markdown()
    assert "\n## fake heading" not in md
    assert "line one ## fake heading" in md
