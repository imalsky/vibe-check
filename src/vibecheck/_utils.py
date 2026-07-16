"""Small helpers shared by the diagnostic checks.

Kept deliberately minimal: anything only one check needs stays in that check.
This module holds what several checks share, namely pulling arrays out of the
orchestrator context, building SKIP results, the optional matplotlib import,
and a numpy Kolmogorov-Smirnov distance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import CheckResult, Status


def skip(name: str, reason: str) -> CheckResult:
    """Build a SKIP result: the check could not run because inputs were absent.

    SKIP is distinct from PASS on purpose, so a report never claims to have
    tested something it did not.
    """
    return CheckResult(name=name, status=Status.SKIP, summary=f"skipped: {reason}")


def get_array(context: dict[str, Any], key: str) -> np.ndarray | None:
    """Return ``context[key]`` as a float ndarray, or None if it is absent."""
    value = context.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def as_2d(a: np.ndarray) -> np.ndarray:
    """Reshape to ``(n_samples, n_features)`` by flattening trailing axes."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


def import_matplotlib():
    """Return ``matplotlib.pyplot``, or None if the ``viz`` extra is not present.

    Checks call this to attach figures when they can and simply omit figures
    when matplotlib is absent, so the core never hard-depends on it. A
    non-interactive backend is selected defensively for headless use.
    """
    try:
        import matplotlib

        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance between two 1-D samples.

    The KS distance is the largest gap between the two empirical CDFs, a value
    in ``[0, 1]`` where larger means the samples differ more. Computed in numpy
    so scipy is not a dependency. Returns NaN if either sample is empty.
    """
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))
