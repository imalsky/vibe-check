"""Calibration checks: are stated uncertainties honest.

If a surrogate reports uncertainty, the empirical coverage should match the
stated level. This module compares nominal vs empirical coverage, in the spirit
of distribution-free / conformal prediction (Angelopoulos & Bates 2021).
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def coverage(**context: Any) -> CheckResult:
    """Compare nominal uncertainty levels against empirical coverage."""
    raise NotImplementedError


coverage.check_name = "calibration.coverage"
