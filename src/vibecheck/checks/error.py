"""Error checks: pointwise and field-level accuracy diagnostics.

Aggregate error hides local failure. These checks produce the plots and tables
that make error legible: predicted-vs-true, residual histograms, per-channel
error tables, and (for spatial-field surrogates) true / predicted / percent-
error maps.
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def pointwise(**context: Any) -> CheckResult:
    """Predicted-vs-true, residuals, and per-channel error summaries."""
    raise NotImplementedError


pointwise.check_name = "error.pointwise"


def field(**context: Any) -> CheckResult:
    """True / predicted / percent-error maps for spatial-field surrogates."""
    raise NotImplementedError


field.check_name = "error.field"
