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
