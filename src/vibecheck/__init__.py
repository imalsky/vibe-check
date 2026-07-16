"""vibe-check: reliability checks for ML surrogates in scientific software.

The public API is intentionally small. The entry point is :func:`check`, which
runs the available diagnostics over a surrogate and its data splits and returns
a :class:`Report`.

This package is in early development; the API is not yet stable.
"""

from __future__ import annotations

from .core import CheckResult, Report, Status, check

__all__ = ["check", "CheckResult", "Report", "Status"]
__version__ = "0.0.1"
