"""Constraint checks: known physics the surrogate must not violate.

Classical solvers respect physical laws by construction; surrogates do not.
These checks test user-declared constraints on the model outputs: conservation
(mass / energy), positivity, monotonicity, symmetry, and hard bounds.
Constraints are supplied through ``metadata`` so the package stays domain
agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import skip
from ..core import CheckResult, Status


def physical(**context: Any) -> CheckResult:
    """Test user-declared physical constraints against the surrogate's outputs.

    Constraints are listed in ``metadata['constraints']``, each a dict with a
    ``type`` and its parameters. Supported types:

    - ``positivity`` / ``nonnegative``: values are >= 0.
    - ``bounds``: ``low`` <= values <= ``high``.
    - ``sum`` / ``conservation``: values sum to ``value`` (default 1.0) along
      ``axis`` (default last), e.g. mass fractions.
    - ``monotonic``: values are non-decreasing (``increasing=True``, default) or
      non-increasing along ``axis``.
    - ``symmetry``: values equal their reverse along ``axis``.

    Every type accepts an absolute ``tol`` (default 1e-6), an optional
    ``channels`` list (with ``channel_axis``, default last) to restrict which
    outputs it applies to, and a ``name`` for the report. The verdict is on the
    worst per-constraint violation fraction: above ``fail_frac`` (default 0.01,
    under ``metadata['constraints_config']``) -> FAIL, any smaller violation ->
    WARN, none -> PASS. Needs ``predict``, ``X_test``, and at least one
    evaluable constraint; SKIPs otherwise.
    """
    name = "constraints.physical"
    metadata = context.get("metadata") or {}
    specs = metadata.get("constraints")
    cfg = metadata.get("constraints_config") or {}
    fail_frac = float(cfg.get("fail_frac", 0.01))
    predict = context.get("predict")
    x_test = context.get("X_test")
    if not specs or predict is None or x_test is None:
        return skip(name, "need predict, X_test, and metadata['constraints']")

    try:
        y_hat = np.asarray(predict(np.asarray(x_test)), dtype=float)
    except Exception as exc:
        return CheckResult(
            name=name, status=Status.FAIL, summary=f"predict raised: {exc!r}"
        )

    metrics: dict[str, float] = {}
    detail_lines = []
    worst_frac = 0.0
    evaluated = 0
    for i, spec in enumerate(specs):
        label = spec.get("name") or f"{spec.get('type', 'constraint')}_{i}"
        result = _eval_constraint(y_hat, spec)
        if result is None:
            detail_lines.append(
                f"- {label}: unknown constraint type '{spec.get('type')}' (skipped)"
            )
            continue
        evaluated += 1
        n_viol, n_total, max_mag = result
        frac = n_viol / n_total if n_total else 0.0
        worst_frac = max(worst_frac, frac)
        metrics[f"{label}_violation_fraction"] = frac
        detail_lines.append(
            f"- {label}: {n_viol}/{n_total} violations "
            f"(fraction {frac:.3g}, max magnitude {max_mag:.3g})"
        )

    if evaluated == 0:
        return skip(name, "no evaluable constraints were provided")

    metrics["max_violation_fraction"] = worst_frac
    details = "\n".join(detail_lines)

    if worst_frac > fail_frac:
        status = Status.FAIL
        summary = f"declared constraints violated (worst {worst_frac:.1%} of values)"
    elif worst_frac > 0:
        status = Status.WARN
        summary = f"minor constraint violations (worst {worst_frac:.2%} of values)"
    else:
        status = Status.PASS
        summary = "all declared physical constraints satisfied"

    return CheckResult(
        name=name, status=status, summary=summary, metrics=metrics, details=details
    )


physical.check_name = "constraints.physical"


def _select(y: np.ndarray, spec: dict) -> np.ndarray:
    """Restrict ``y`` to the constraint's channels, if any were named."""
    channels = spec.get("channels")
    if channels is None:
        return y
    return np.take(y, np.asarray(channels), axis=int(spec.get("channel_axis", -1)))


def _eval_constraint(y: np.ndarray, spec: dict):
    """Return ``(n_violations, n_total, max_magnitude)`` or None if unknown."""
    ctype = spec.get("type")
    tol = float(spec.get("tol", 1e-6))

    if ctype in ("positivity", "nonnegative"):
        vals = _select(y, spec)
        magnitude = np.clip(-vals, 0.0, None)
        return int(np.sum(vals < -tol)), int(vals.size), _max0(magnitude)

    if ctype == "bounds":
        vals = _select(y, spec)
        low = float(spec.get("low", -np.inf))
        high = float(spec.get("high", np.inf))
        excess = np.clip(np.maximum(low - vals, vals - high), 0.0, None)
        return int(np.sum(excess > tol)), int(vals.size), _max0(excess)

    if ctype in ("sum", "conservation"):
        axis = int(spec.get("axis", -1))
        value = float(spec.get("value", 1.0))
        residual = np.abs(_select(y, spec).sum(axis=axis) - value)
        return int(np.sum(residual > tol)), int(residual.size), _max0(residual)

    if ctype in ("monotonic", "monotonicity"):
        axis = int(spec.get("axis", -1))
        increasing = bool(spec.get("increasing", True))
        diffs = np.diff(_select(y, spec), axis=axis)
        signed = -diffs if increasing else diffs
        magnitude = np.clip(signed, 0.0, None)
        return int(np.sum(signed > tol)), int(diffs.size), _max0(magnitude)

    if ctype == "symmetry":
        axis = int(spec.get("axis", -1))
        vals = _select(y, spec)
        diff = np.abs(vals - np.flip(vals, axis=axis))
        return int(np.sum(diff > tol)), int(vals.size), _max0(diff)

    return None


def _max0(a: np.ndarray) -> float:
    return float(a.max()) if a.size else 0.0
