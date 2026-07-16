"""Core types and the top-level orchestrator.

This module defines the shared contract (``Status``, ``CheckResult``, ``Report``)
and the :func:`check` entry point. The individual diagnostics live in
``vibecheck.checks`` and are wired in here as they are implemented.
"""

from __future__ import annotations

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
        order = {"pass": 0, "skip": 0, "warn": 1, "fail": 2, "error": 3}
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


@dataclass
class Report:
    """Aggregates :class:`CheckResult` objects and renders them."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def summary(self) -> Status:
        """Worst status across all checks. Useful for gating CI."""
        if not self.results:
            return Status.SKIP
        return max((r.status for r in self.results), key=lambda s: s.rank)

    def to_markdown(self, path: str | None = None) -> str:
        lines = ["# vibe-check report", "", f"Overall: **{self.summary().value.upper()}**", ""]
        for r in self.results:
            lines.append(f"## {r.name} - {r.status.value.upper()}")
            lines.append(r.summary)
            if r.metrics:
                lines.append("")
                for k, v in r.metrics.items():
                    lines.append(f"- `{k}`: {v}")
            if r.details:
                lines.append("")
                lines.append(r.details)
            lines.append("")
        text = "\n".join(lines)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def to_html(self, path: str | None = None) -> str:
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
        overall_color = _STATUS_COLORS.get(overall, "#000000")
        parts = [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>vibe-check report</title>",
            "<style>",
            "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
            "max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5;}",
            ".status{font-weight:600;text-transform:uppercase;}",
            ".check{border:1px solid #d0d7de;border-radius:6px;"
            "padding:0.5rem 1rem;margin:1rem 0;}",
            ".metrics{font-family:ui-monospace,monospace;font-size:0.9em;}",
            "img{max-width:100%;height:auto;}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>vibe-check report</h1>",
            f'<p>Overall: <span class="status" style="color:{overall_color}">'
            f"{html.escape(overall.upper())}</span></p>",
        ]

        if not self.results:
            parts.append("<p>No checks were run.</p>")

        for r in self.results:
            color = _STATUS_COLORS.get(r.status.value, "#000000")
            parts.append('<section class="check">')
            parts.append(
                f'<h2>{html.escape(r.name)} '
                f'<span class="status" style="color:{color}">'
                f"{html.escape(r.status.value.upper())}</span></h2>"
            )
            parts.append(f"<p>{html.escape(r.summary)}</p>")
            if r.metrics:
                parts.append('<ul class="metrics">')
                for k, v in r.metrics.items():
                    parts.append(f"<li>{html.escape(str(k))}: {html.escape(str(v))}</li>")
                parts.append("</ul>")
            if r.details:
                parts.append(f"<p>{html.escape(r.details)}</p>")
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


# Status value -> CSS color, shared by the HTML renderer.
_STATUS_COLORS = {
    "pass": "#1a7f37",
    "warn": "#9a6700",
    "fail": "#cf222e",
    "skip": "#6e7781",
    "error": "#cf222e",
}


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


# The registry of checks the orchestrator will run. Each implemented check
# appends itself here as it lands. Kept empty until the first check is wired in
# so that `check()` is honest about running nothing yet.
_REGISTERED_CHECKS: list[Callable[..., CheckResult]] = []


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
    """Run the available diagnostics over a surrogate and return a Report.

    Parameters mirror ``docs/VALIDATION_CONTRACT.md``. ``predict`` maps inputs
    to predicted outputs in batches. The data splits and ``metadata`` are
    optional; checks that need inputs they were not given return ``Status.SKIP``.

    This is a skeleton: no checks are registered yet, so the returned report is
    empty. Implementing the checks in ``vibecheck.checks`` and registering them
    is the next milestone (see ``ROADMAP.md``).
    """
    report = Report()
    context = dict(
        predict=predict,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        metadata=metadata or {},
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
    """Populate the registry with the checks that are implemented.

    Imported here, at module load, rather than at the top of the file: the
    check modules import ``CheckResult`` from this module, so importing them at
    the top would create a cycle. A check is added to this list in the same
    change that implements and tests it, so the registry never holds a stub
    that would raise.
    """
    from .checks import distribution, leakage

    _REGISTERED_CHECKS.extend(
        [
            leakage.normalization,
            leakage.split_overlap,
            distribution.coverage,
        ]
    )


_register_default_checks()
