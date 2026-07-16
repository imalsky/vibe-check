"""Calibration checks: are stated uncertainties honest.

If a surrogate reports uncertainty, the empirical coverage should match the
stated level. This module compares nominal vs empirical coverage, in the spirit
of distribution-free / conformal prediction (Angelopoulos & Bates 2021).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .._utils import get_array, skip, want_figures
from ..core import CheckResult, Status


def coverage(**context: Any) -> CheckResult:
    """Compare a surrogate's stated uncertainty against its empirical coverage.

    The surrogate must supply a predictive standard deviation, either by having
    ``predict`` return a ``(mean, std)`` tuple, or via
    ``metadata['predicted_std']`` (an array aligned with ``y_test``, or a single
    scalar). Under a Gaussian reading, the check measures how often the true
    value falls within k standard deviations of the mean for k in {1, 2, 3} and
    compares that to the nominal Gaussian coverage (68.3, 95.5, 99.7 percent).

    The verdict is on the largest gap between empirical and nominal coverage:
    above ``fail_tol`` (default 0.15) -> FAIL, above ``warn_tol`` (default
    0.07) -> WARN, otherwise PASS. Empirical coverage below nominal means the
    intervals are too narrow (overconfident); above means too wide. Thresholds
    live under ``metadata['calibration']``. SKIPs when no uncertainty is given.
    Set ``metadata['make_figures'] = True`` for a reliability plot.
    """
    name = "calibration.coverage"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("calibration") or {}
    predict = context.get("predict")
    y_test = get_array(context, "y_test")
    x_test = context.get("X_test")
    if predict is None or y_test is None or x_test is None:
        return skip(name, "need predict, X_test, and y_test")

    y_true = np.asarray(y_test, dtype=float)
    out = predict(np.asarray(x_test))
    if isinstance(out, (tuple, list)) and len(out) == 2:
        mean = np.asarray(out[0], dtype=float)
        std = np.asarray(out[1], dtype=float)
    else:
        mean = np.asarray(out, dtype=float)
        provided = metadata.get("predicted_std", metadata.get("uncertainty"))
        std = None if provided is None else np.asarray(provided, dtype=float)
    if std is None:
        return skip(
            name,
            "no predicted uncertainty (return (mean, std) or set "
            "metadata['predicted_std'])",
        )

    if mean.shape != y_true.shape and mean.size == y_true.size:
        mean = mean.reshape(y_true.shape)
    if std.shape != y_true.shape:
        if std.size == 1:
            std = np.full(y_true.shape, float(std))
        elif std.size == y_true.size:
            std = std.reshape(y_true.shape)
        else:
            return skip(name, "predicted uncertainty shape does not match y_test")
    std = np.abs(std) + 1e-12

    z = np.abs(y_true - mean) / std
    ks = (1.0, 2.0, 3.0)
    empirical = {k: float(np.mean(z <= k)) for k in ks}
    nominal = {k: math.erf(k / math.sqrt(2.0)) for k in ks}
    signed = {k: empirical[k] - nominal[k] for k in ks}
    max_dev = max(abs(d) for d in signed.values())
    direction = (
        "overconfident (intervals too narrow)"
        if signed[1.0] < 0
        else "underconfident (intervals too wide)"
    )

    metrics = {
        "empirical_coverage_1sigma": empirical[1.0],
        "empirical_coverage_2sigma": empirical[2.0],
        "empirical_coverage_3sigma": empirical[3.0],
        "nominal_coverage_1sigma": nominal[1.0],
        "nominal_coverage_2sigma": nominal[2.0],
        "max_abs_coverage_deviation": max_dev,
    }

    figures = []
    plt = want_figures(context)
    if plt is not None:
        sweep = np.linspace(0.1, 3.0, 25)
        nom_curve = [math.erf(k / math.sqrt(2.0)) for k in sweep]
        emp_curve = [float(np.mean(z <= k)) for k in sweep]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
        ax.plot(nom_curve, emp_curve, color="#1a6acf")
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical coverage")
        ax.set_title("calibration.coverage: reliability")
        fig.tight_layout()
        figures.append(fig)

    warn_tol = float(cfg.get("warn_tol", 0.07))
    fail_tol = float(cfg.get("fail_tol", 0.15))
    if max_dev > fail_tol:
        status = Status.FAIL
        summary = f"stated uncertainty is miscalibrated (max gap {max_dev:.2f}); {direction}"
    elif max_dev > warn_tol:
        status = Status.WARN
        summary = f"stated uncertainty is loosely calibrated (max gap {max_dev:.2f}); {direction}"
    else:
        status = Status.PASS
        summary = f"stated uncertainty is well calibrated (max gap {max_dev:.2f})"

    return CheckResult(
        name=name, status=status, summary=summary, metrics=metrics, figures=figures
    )


coverage.check_name = "calibration.coverage"
