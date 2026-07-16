"""Export checks: does the saved model reproduce the in-memory one.

A surrogate that is accurate in a notebook but diverges once exported to ONNX /
TorchScript / a saved checkpoint is not reproducible. This check runs the same
inputs through both paths and reports the largest discrepancy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import skip
from ..core import CheckResult, Status


def roundtrip(**context: Any) -> CheckResult:
    """Compare an exported model's outputs against the in-memory model.

    Provide the exported model as a second callable in
    ``metadata['exported_predict']`` (wrap ONNXRuntime, TorchScript, or a
    reloaded checkpoint so it maps ``X -> y_hat`` like ``predict``). The check
    runs ``X_test`` through both and compares them elementwise with numpy's
    ``isclose`` at absolute tolerance ``atol`` (default 1e-4) and relative
    tolerance ``rtol`` (default 1e-3), both under ``metadata['export']``.

    Verdict: every element within tolerance -> PASS; a small fraction differing
    (at or below ``warn_frac``, default 0.01) -> WARN; more than that -> FAIL. A
    shape mismatch between the two paths is a FAIL, and a raised call on either
    path is a FAIL naming which path raised. Needs ``predict``, ``X_test``, and
    ``metadata['exported_predict']``; SKIPs otherwise.
    """
    name = "export.roundtrip"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("export") or {}
    predict = context.get("predict")
    exported = metadata.get("exported_predict")
    x_test = context.get("X_test")
    if predict is None or exported is None or x_test is None:
        return skip(name, "need predict, X_test, and metadata['exported_predict']")

    atol = float(cfg.get("atol", 1e-4))
    rtol = float(cfg.get("rtol", 1e-3))
    warn_frac = float(cfg.get("warn_frac", 0.01))

    x = np.asarray(x_test)
    try:
        a = np.asarray(predict(x), dtype=float)
    except Exception as exc:
        return CheckResult(
            name=name,
            status=Status.FAIL,
            summary=f"in-memory predict raised: {exc!r}",
        )
    try:
        b = np.asarray(exported(x), dtype=float)
    except Exception as exc:
        return CheckResult(
            name=name,
            status=Status.FAIL,
            summary=f"exported predict raised: {exc!r}",
        )

    if a.shape != b.shape:
        return CheckResult(
            name=name,
            status=Status.FAIL,
            summary=(
                f"in-memory output shape {a.shape} does not match exported "
                f"output shape {b.shape}"
            ),
        )

    diff = np.abs(a - b)
    close = np.isclose(a, b, atol=atol, rtol=rtol)
    frac_fail = float(1.0 - close.mean())
    max_abs = float(diff.max()) if diff.size else 0.0
    scale = np.abs(b) + atol
    max_rel = float((diff / scale).max()) if diff.size else 0.0

    metrics = {
        "max_abs_difference": max_abs,
        "max_rel_difference": max_rel,
        "fraction_mismatched": frac_fail,
        "atol": atol,
        "rtol": rtol,
    }

    if frac_fail <= 0.0:
        status = Status.PASS
        summary = f"exported model matches in-memory model (max abs diff {max_abs:.3g})"
    elif frac_fail <= warn_frac:
        status = Status.WARN
        summary = (
            f"{frac_fail:.2%} of outputs differ beyond tolerance "
            f"(max abs diff {max_abs:.3g})"
        )
    else:
        status = Status.FAIL
        summary = (
            f"exported model diverges from in-memory model: "
            f"{frac_fail:.1%} of outputs differ (max abs diff {max_abs:.3g})"
        )

    return CheckResult(name=name, status=status, summary=summary, metrics=metrics)


roundtrip.check_name = "export.roundtrip"
