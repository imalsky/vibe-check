"""Leakage checks: normalization hygiene and train/test overlap.

Data leakage is the single most common way a scientific ML result is silently
inflated (Kapoor & Narayanan 2023). Two concrete failures to detect:

- ``normalization``: the scaler / per-channel statistics were fit on the test
  set or on the full dataset instead of on training data only.
- ``split_overlap``: identical or near-identical rows appear in more than one
  split, so the test set is not actually held out.
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def normalization(**context: Any) -> CheckResult:
    """Detect normalization statistics that were not fit on train-only data."""
    raise NotImplementedError


normalization.check_name = "leakage.normalization"


def split_overlap(**context: Any) -> CheckResult:
    """Detect duplicate / near-duplicate samples shared across data splits."""
    raise NotImplementedError


split_overlap.check_name = "leakage.split_overlap"
