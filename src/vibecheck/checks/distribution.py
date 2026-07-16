"""Distribution checks: training-domain coverage and split drift.

A surrogate is only trustworthy inside the region it was trained on. These
checks compare the train / validation / test input (and output) distributions
and flag test points that sit outside the training domain, where the model is
extrapolating rather than interpolating.
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def coverage(**context: Any) -> CheckResult:
    """Flag test inputs that fall outside the training domain (extrapolation)."""
    raise NotImplementedError


coverage.check_name = "distribution.coverage"


def drift(**context: Any) -> CheckResult:
    """Compare train/val/test marginals (histograms, Q-Q, KS distance)."""
    raise NotImplementedError


drift.check_name = "distribution.drift"
