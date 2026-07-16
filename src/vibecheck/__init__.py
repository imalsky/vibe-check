"""vibe-check: reliability checks for ML surrogates in scientific software.

The public API is intentionally small. The entry point is :func:`check`, which
runs the available diagnostics over a surrogate and its data splits and returns
a :class:`Report`.

The API is close to stable but may still change before 1.0.
"""

from __future__ import annotations

from .core import CheckResult, Report, Status, check

__all__ = ["check", "CheckResult", "Report", "Status"]
__version__ = "0.1.0"
