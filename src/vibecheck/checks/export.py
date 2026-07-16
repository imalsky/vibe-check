"""Export checks: does the saved model reproduce the in-memory one.

A surrogate that is accurate in a notebook but diverges once exported to ONNX /
TorchScript / a saved checkpoint is not reproducible. This check runs the same
inputs through both paths and reports the largest discrepancy.
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def roundtrip(**context: Any) -> CheckResult:
    """Compare exported-model outputs against the in-memory model outputs."""
    raise NotImplementedError


roundtrip.check_name = "export.roundtrip"
