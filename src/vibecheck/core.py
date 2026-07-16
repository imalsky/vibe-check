"""Core types and the top-level orchestrator.

This module defines the shared contract (``Status``, ``CheckResult``, ``Report``)
and the :func:`check` entry point. The individual diagnostics live in
``vibecheck.checks`` and are registered here at import time.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    """Outcome of a single check.

    ``SKIP`` (not applicable / inputs missing) and ``ERROR`` (the check itself
    raised) are deliberately distinct from ``PASS`` so a report never claims to
    have tested something it did not.
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Severity order used to aggregate statuses (higher is worse).

        ``SKIP`` ranks below ``PASS``: a report with one passed check and nine
        skipped checks passed something, so its overall status is ``PASS``, not
        ``SKIP``.
        """
        order = {"skip": 0, "pass": 1, "warn": 2, "fail": 3, "error": 4}
        return order[self.value]


@dataclass
class CheckResult:
    """The result of one diagnostic. See ``docs/VALIDATION_CONTRACT.md``."""

    name: str
    status: Status
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: str = ""
    figures: list[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}: {self.status.value.upper()} - {self.summary}"


@dataclass
class Report:
    """Aggregates :class:`CheckResult` objects and renders them."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def summary(self) -> Status:
        """Worst status across all checks. Useful for gating CI.

        An empty report returns ``SKIP``. A report where anything passed and
        nothing warned, failed, or errored returns ``PASS``; ``SKIP`` is the
        overall status only when every check skipped.
        """
        if not self.results:
            return Status.SKIP
        return max((r.status for r in self.results), key=lambda s: s.rank)

    def __str__(self) -> str:
        lines = [f"vibe-check: {self.summary().value.upper()}"]
        lines.extend(f"  {r}" for r in self.results)
        return "\n".join(lines)

    def _worst_first(self) -> list[CheckResult]:
        return sorted(self.results, key=lambda r: r.status.rank, reverse=True)

    def to_markdown(self, path: str | None = None, *, title: str = "vibe-check report") -> str:
        """Render the report as markdown and return it (optionally writing ``path``).

        The report opens with the overall status and a worst-first summary
        table, followed by one section per check in the order the checks ran.
        ``details`` is emitted as raw markdown; figures are only rendered by
        :meth:`to_html`, so a check with figures gets a pointer line here.
        Metric values are formatted for display; the full-precision numbers
        stay available on each result's ``metrics`` dict.
        """
        lines = [f"# {_oneline(title)}", "", f"Overall: **{self.summary().value.upper()}**", ""]
        if not self.results:
            lines.append("No checks were run.")
            lines.append("")
        else:
            lines.append("| check | status | summary |")
            lines.append("| --- | --- | --- |")
            for r in self._worst_first():
                lines.append(
                    f"| {_oneline(r.name)} | {r.status.value.upper()} | {_oneline(r.summary)} |"
                )
            lines.append("")
        for r in self.results:
            lines.append(f"## {_oneline(r.name)} - {r.status.value.upper()}")
            lines.append(_oneline(r.summary))
            if r.metrics:
                lines.append("")
                for k, v in r.metrics.items():
                    lines.append(f"- `{k}`: {_format_metric(v)}")
            if r.details:
                lines.append("")
                lines.append(r.details)
            if r.figures:
                lines.append("")
                n = len(r.figures)
                lines.append(f"({n} figure{'s' if n != 1 else ''} attached; see the HTML report)")
            lines.append("")
        text = "\n".join(lines)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_html(self, path: str | None = None, *, title: str = "vibe-check report") -> str:
        """Render the report as a single self-contained HTML document.

        Figures are embedded as base64 PNG so the file has no external
        dependencies. Any figure object exposing ``savefig(buffer, format=...)``
        (such as a matplotlib ``Figure``) is supported; matplotlib itself is
        never imported here, so the core stays numpy-only. A figure that cannot
        be rendered is skipped with a visible note rather than raising, and all
        text is HTML-escaped.
        """
        import html

        overall = self.summary().value
        safe_title = html.escape(_oneline(title))
        parts = [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{safe_title}</title>",
            "<style>",
            ":root{color-scheme:light;}",
            "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
            "max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5;"
            "background:#ffffff;color:#1f2328;}",
            "h1{font-size:1.6rem;}",
            ".status{display:inline-block;font-weight:600;text-transform:uppercase;"
            "font-size:0.78em;padding:0.1em 0.6em;border-radius:2em;vertical-align:middle;}",
            _status_css(),
            "table.overview{border-collapse:collapse;width:100%;margin:1rem 0;}",
            "table.overview th,table.overview td{border:1px solid #d0d7de;"
            "padding:0.35rem 0.6rem;text-align:left;font-size:0.95em;}",
            "table.overview th{background:#f6f8fa;}",
            "table.overview a{color:inherit;text-decoration:none;}",
            "table.overview a:hover{text-decoration:underline;}",
            ".check{border:1px solid #d0d7de;border-radius:6px;"
            "padding:0.5rem 1rem;margin:1rem 0;}",
            ".check h2{font-size:1.15rem;margin:0.4rem 0 0.6rem;}",
            ".metrics{font-family:ui-monospace,monospace;font-size:0.9em;"
            "overflow-wrap:anywhere;}",
            ".details{white-space:pre-line;}",
            "img{max-width:100%;height:auto;}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{safe_title}</h1>",
            f'<p>Overall: <span class="status st-{html.escape(overall)}">'
            f"{html.escape(overall.upper())}</span></p>",
        ]

        if not self.results:
            parts.append("<p>No checks were run.</p>")
        else:
            parts.append('<table class="overview">')
            parts.append("<tr><th>check</th><th>status</th><th>summary</th></tr>")
            for r in self._worst_first():
                status = html.escape(r.status.value)
                parts.append(
                    f'<tr><td><a href="#{_anchor(r.name)}">{html.escape(r.name)}</a></td>'
                    f'<td><span class="status st-{status}">{status.upper()}</span></td>'
                    f"<td>{html.escape(_oneline(r.summary))}</td></tr>"
                )
            parts.append("</table>")

        for r in self.results:
            status = html.escape(r.status.value)
            parts.append(f'<section class="check" id="{_anchor(r.name)}">')
            parts.append(
                f"<h2>{html.escape(r.name)} "
                f'<span class="status st-{status}">{status.upper()}</span></h2>'
            )
            parts.append(f"<p>{html.escape(r.summary)}</p>")
            if r.metrics:
                parts.append('<ul class="metrics">')
                for k, v in r.metrics.items():
                    parts.append(f"<li>{html.escape(str(k))}: {_format_metric(v)}</li>")
                parts.append("</ul>")
            if r.details:
                parts.append(f'<p class="details">{html.escape(r.details)}</p>')
            for fig in r.figures:
                encoded = _encode_figure(fig)
                if encoded is None:
                    parts.append("<p><em>(figure could not be rendered)</em></p>")
                else:
                    parts.append(
                        f'<img alt="figure for {html.escape(r.name)}" '
                        f'src="data:image/png;base64,{encoded}">'
                    )
            parts.append("</section>")

        parts.append("</body>")
        parts.append("</html>")
        text = "\n".join(parts)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text


# Status value -> (text color, badge background), shared by the HTML renderer.
# ERROR gets its own color so a crashed check is visually distinct from a
# check that ran and failed.
_STATUS_COLORS = {
    "pass": ("#1a7f37", "#dafbe1"),
    "warn": ("#9a6700", "#fff8c5"),
    "fail": ("#cf222e", "#ffebe9"),
    "skip": ("#57606a", "#f6f8fa"),
    "error": ("#8250df", "#fbefff"),
}


def _status_css() -> str:
    return "".join(
        f".st-{name}{{color:{fg};background:{bg};}}"
        for name, (fg, bg) in _STATUS_COLORS.items()
    )


def _anchor(name: str) -> str:
    return "check-" + "".join(c if c.isalnum() else "-" for c in name.lower())


def _oneline(text: str) -> str:
    """Collapse whitespace runs and newlines: these fields are one line by contract."""
    return " ".join(str(text).split())


def _format_metric(value: Any) -> str:
    """Format a metric for display: ints without '.0', floats to 4 significant digits.

    Display-only; the machine-readable ``metrics`` dict keeps full precision.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _oneline(value)
    if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.4g}"


def _encode_figure(fig: Any) -> str | None:
    """Encode a figure as a base64 PNG string, or return None if it cannot be.

    Duck-typed on ``savefig`` so matplotlib is not imported by the core. Any
    rendering failure is swallowed and reported as None so one bad figure never
    sinks a report.
    """
    savefig = getattr(fig, "savefig", None)
    if savefig is None:
        return None
    import base64
    import io

    try:
        buffer = io.BytesIO()
        savefig(buffer, format="png", bbox_inches="tight")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


# The registry of checks the orchestrator runs, populated at import time by
# _register_default_checks() below. Each check was added in the same change
# that implemented and tested it, so the registry never holds a stub.
_REGISTERED_CHECKS: list[Callable[..., CheckResult]] = []


# Top-level metadata keys read by the registered checks; used to warn on typos.
# Kept in sync with the table in docs/VALIDATION_CONTRACT.md.
_KNOWN_METADATA_KEYS = frozenset(
    {
        "normalization",
        "split_overlap",
        "coverage",
        "drift",
        "error",
        "error_field",
        "predicted_std",
        "calibration",
        "constraints",
        "constraints_config",
        "exported_predict",
        "export",
        "speed",
        "make_figures",
    }
)


def _warn_unknown_metadata_keys(metadata: dict[str, Any]) -> None:
    unknown = sorted(set(metadata) - _KNOWN_METADATA_KEYS)
    if not unknown:
        return
    import difflib

    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, _KNOWN_METADATA_KEYS, n=1)
        hints.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    warnings.warn(
        "unrecognized metadata key(s) ignored by all checks: "
        + ", ".join(hints)
        + "; see the metadata table in docs/VALIDATION_CONTRACT.md",
        stacklevel=3,
    )


def check(
    predict: Callable[[Any], Any],
    *,
    X_train: Any = None,
    y_train: Any = None,
    X_val: Any = None,
    y_val: Any = None,
    X_test: Any = None,
    y_test: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Report:
    """Run the registered diagnostics over a surrogate and return a Report.

    Parameters mirror ``docs/VALIDATION_CONTRACT.md``. ``predict`` maps a batch
    of inputs to a batch of predicted outputs (numpy in, numpy out). The data
    splits and ``metadata`` are optional; every registered check runs, and a
    check whose required inputs were not given returns ``Status.SKIP`` rather
    than passing silently. A check that raises is recorded as ``Status.ERROR``
    without sinking the rest of the run. Unrecognized top-level ``metadata``
    keys trigger a ``UserWarning`` so a typo cannot silently disable a check.
    """
    report = Report()
    metadata = metadata or {}
    _warn_unknown_metadata_keys(metadata)
    context = dict(
        predict=predict,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metadata=metadata,
    )
    for check_fn in _REGISTERED_CHECKS:
        try:
            report.add(check_fn(**context))
        except Exception as exc:  # a broken check must not sink the whole run
            report.add(
                CheckResult(
                    name=getattr(check_fn, "check_name", check_fn.__name__),
                    status=Status.ERROR,
                    summary=f"check raised: {exc!r}",
                )
            )
    return report


def _register_default_checks() -> None:
    """Populate the registry with the ten implemented checks.

    Imported here, at module load, rather than at the top of the file: the
    check modules import ``CheckResult`` from this module, so importing them at
    the top would create a cycle.
    """
    from .checks import (
        calibration,
        constraints,
        distribution,
        error,
        export,
        leakage,
        speed,
    )

    _REGISTERED_CHECKS.extend(
        [
            leakage.normalization,
            leakage.split_overlap,
            distribution.coverage,
            distribution.drift,
            error.pointwise,
            error.field,
            calibration.coverage,
            constraints.physical,
            export.roundtrip,
            speed.inference,
        ]
    )


_register_default_checks()
