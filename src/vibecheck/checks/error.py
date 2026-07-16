"""Error checks: pointwise and field-level accuracy diagnostics.

Aggregate error hides local failure. These checks produce the numbers and plots
that make error legible: predicted-vs-true, residual histograms, per-channel
error tables, and (for spatial-field surrogates) true / predicted / percent-
error maps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import get_array, skip, want_figures
from ..core import CheckResult, Status


def _squeezed(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return ``shape`` with size-1 axes removed, e.g. ``(n, 1)`` -> ``(n,)``."""
    return tuple(d for d in shape if d != 1)


def pointwise(**context: Any) -> CheckResult:
    """Predicted-vs-true error, with a skill score against a mean baseline.

    Runs ``predict`` on ``X_test`` and compares to ``y_test``, reporting RMSE,
    MAE, max absolute error, R^2, and a skill score relative to the naive
    baseline of always predicting the (per-channel) training mean. A useful
    surrogate must beat that baseline.

    Verdict:

    - predictions with NaN/inf, a shape mismatch, or a raised ``predict`` -> FAIL;
    - skill at or below zero (no better than predicting the mean) -> FAIL;
    - an RMSE budget ``metadata['error']['max_rmse']`` that is exceeded -> FAIL;
    - skill undefined because the target is constant -> WARN;
    - positive but low skill (below ``warn_skill``, default 0.5) -> WARN;
    - otherwise PASS.

    A prediction is only reshaped to ``y_test``'s shape when the two shapes
    agree after dropping size-1 axes; any other mismatch is a FAIL rather than
    a silent reshape. The training-mean baseline is used only when ``y_train``
    has the same per-sample shape as ``y_test``; otherwise skill falls back to
    R^2 and the details say so. Effective thresholds are echoed in the metrics.

    Needs ``predict``, ``X_test``, and ``y_test``; SKIPs otherwise. Set
    ``metadata['make_figures'] = True`` for a predicted-vs-true scatter and a
    residual histogram (built from the finite values only).
    """
    name = "error.pointwise"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("error") or {}
    predict = context.get("predict")
    y_test = get_array(context, "y_test")
    x_test = context.get("X_test")
    if predict is None or y_test is None or x_test is None:
        return skip(name, "need predict, X_test, and y_test")

    y_true = np.asarray(y_test, dtype=float)
    try:
        y_hat = np.asarray(predict(np.asarray(x_test)), dtype=float)
    except Exception as exc:
        return CheckResult(
            name=name, status=Status.FAIL, summary=f"predict raised: {exc!r}"
        )

    if y_hat.shape != y_true.shape:
        if _squeezed(y_hat.shape) == _squeezed(y_true.shape):
            y_hat = y_hat.reshape(y_true.shape)
        else:
            return CheckResult(
                name=name,
                status=Status.FAIL,
                summary=(
                    f"prediction shape {y_hat.shape} does not match "
                    f"y_test shape {y_true.shape}"
                ),
            )

    all_finite = bool(np.isfinite(y_hat).all())
    yt = y_true.ravel()
    yp = y_hat.ravel()
    resid = yp - yt
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    max_abs = float(np.max(np.abs(resid))) if resid.size else 0.0
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    detail_lines = []
    y_train = get_array(context, "y_train")
    if y_train is not None and _squeezed(y_train.shape[1:]) == _squeezed(
        y_true.shape[1:]
    ):
        base = y_train.reshape(-1, *y_true.shape[1:]).mean(axis=0)
        rmse_base = float(np.sqrt(np.mean((y_true - base) ** 2)))
        skill = 1.0 - rmse / rmse_base if rmse_base > 0 else float("nan")
    else:
        if y_train is not None:
            detail_lines.append(
                f"y_train per-sample shape {y_train.shape[1:]} does not match "
                f"y_test per-sample shape {y_true.shape[1:]}; skill falls back "
                "to R^2 instead of the training-mean baseline"
            )
        skill = r2

    max_rmse = cfg.get("max_rmse")
    warn_skill = float(cfg.get("warn_skill", 0.5))
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "r2": r2,
        "skill_vs_mean_baseline": skill,
        "warn_skill": warn_skill,
    }
    if max_rmse is not None:
        metrics["max_rmse"] = float(max_rmse)

    n_ch = int(np.prod(y_true.shape[1:])) if y_true.ndim >= 2 else 1
    if 1 < n_ch <= 20:
        per_ch = np.sqrt(
            (
                (y_hat.reshape(y_true.shape[0], n_ch) - y_true.reshape(y_true.shape[0], n_ch))
                ** 2
            ).mean(axis=0)
        )
        detail_lines.append("Per-channel RMSE:")
        detail_lines += [f"- channel {i}: {v:.4g}" for i, v in enumerate(per_ch)]
    details = "\n".join(detail_lines)

    figures = []
    plt = want_figures(context)
    if plt is not None and resid.size:
        # Figures are built from the finite values only, so non-finite
        # predictions still reach the FAIL verdict instead of crashing
        # matplotlib on infinite axis limits or histogram ranges.
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt_f, yp_f, resid_f = yt[finite], yp[finite], resid[finite]
        if yt_f.size:
            rng = np.random.default_rng(0)
            idx = (
                rng.choice(yt_f.size, size=2000, replace=False)
                if yt_f.size > 2000
                else slice(None)
            )
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.scatter(yt_f[idx], yp_f[idx], s=6, alpha=0.4, color="C0")
            lims = [
                float(min(yt_f.min(), yp_f.min())),
                float(max(yt_f.max(), yp_f.max())),
            ]
            ax1.plot(lims, lims, "--", color="#888888", lw=1)
            ax1.set_xlabel("true")
            ax1.set_ylabel("predicted")
            ax1.set_title("error.pointwise: predicted vs true")
            fig1.tight_layout()
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.hist(resid_f, bins=40, color="C0")
            ax2.set_xlabel("residual (predicted - true)")
            ax2.set_ylabel("count")
            ax2.set_title("error.pointwise: residuals")
            fig2.tight_layout()
            figures += [fig1, fig2]

    skill_val = skill if np.isfinite(skill) else r2

    if not all_finite:
        status = Status.FAIL
        summary = "predictions contain NaN or inf"
    elif max_rmse is not None and rmse > float(max_rmse):
        status = Status.FAIL
        summary = f"RMSE {rmse:.4g} exceeds the budget {float(max_rmse):.4g}"
    elif not np.isfinite(skill_val):
        status = Status.WARN
        summary = f"RMSE {rmse:.4g}; skill undefined (constant target)"
    elif skill_val <= 0:
        status = Status.FAIL
        summary = f"no skill: RMSE {rmse:.4g} is no better than predicting the mean"
    elif skill_val < warn_skill:
        status = Status.WARN
        summary = f"low skill {skill_val:.2f} (RMSE {rmse:.4g})"
    else:
        status = Status.PASS
        summary = f"RMSE {rmse:.4g}, skill {skill_val:.2f}"

    return CheckResult(
        name=name,
        status=status,
        summary=summary,
        metrics=metrics,
        details=details,
        figures=figures,
    )


pointwise.check_name = "error.pointwise"


def field(**context: Any) -> CheckResult:
    """Field-level error for spatial-field surrogates: percent-error maps.

    For surrogates whose output is a spatial field (a 1-D profile, a 2-D map,
    etc.) aggregate error is especially misleading, because a low mean can hide
    a localized failure. This check reports the field RMSE, the mean and max
    absolute percent error (per sample, normalized by that sample's mean
    magnitude), the worst sample, and a skill score against the mean-field
    baseline.

    A field is assumed when the per-sample output has at least
    ``metadata['error_field']['min_field_size']`` elements (default 16); set
    ``metadata['error_field']['is_field']`` to force it on or off. Small
    tabular outputs SKIP here (``error.pointwise`` covers them).

    Verdict: NaN/inf or shape mismatch -> FAIL; no skill vs the mean field ->
    FAIL; mean percent error above ``fail_pct`` (default 20) -> FAIL; above
    ``warn_pct`` (default 5) -> WARN; otherwise PASS. Predictions are only
    reshaped to ``y_test``'s shape when the two shapes agree after dropping
    size-1 axes; any other mismatch is a FAIL. The mean-field baseline uses
    ``y_train`` only when its per-sample shape matches ``y_test``; otherwise
    the baseline is derived from ``y_test`` itself and the details say so.
    Effective thresholds are echoed in the metrics. With
    ``metadata['make_figures'] = True`` a side-by-side true / predicted /
    percent-error panel for the worst sample (largest per-sample percent
    error) is attached; the figure is skipped when predictions are not finite.
    """
    name = "error.field"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("error_field") or {}
    predict = context.get("predict")
    y_test = get_array(context, "y_test")
    x_test = context.get("X_test")
    if predict is None or y_test is None or x_test is None:
        return skip(name, "need predict, X_test, and y_test")

    y_true = np.asarray(y_test, dtype=float)
    field_shape = y_true.shape[1:]
    field_size = int(np.prod(field_shape)) if field_shape else 0
    min_field = int(cfg.get("min_field_size", 16))
    is_field = bool(cfg.get("is_field", field_size >= min_field))
    if not is_field:
        return skip(name, f"output is not a spatial field (field size {field_size})")

    try:
        y_hat = np.asarray(predict(np.asarray(x_test)), dtype=float)
    except Exception as exc:
        return CheckResult(
            name=name, status=Status.FAIL, summary=f"predict raised: {exc!r}"
        )
    if y_hat.shape != y_true.shape:
        if _squeezed(y_hat.shape) == _squeezed(y_true.shape):
            y_hat = y_hat.reshape(y_true.shape)
        else:
            return CheckResult(
                name=name,
                status=Status.FAIL,
                summary=(
                    f"prediction shape {y_hat.shape} does not match "
                    f"y_test shape {y_true.shape}"
                ),
            )

    all_finite = bool(np.isfinite(y_hat).all())
    axes = tuple(range(1, y_true.ndim))
    per_sample_mae = np.abs(y_hat - y_true).mean(axis=axes)
    denom = np.abs(y_true).mean(axis=axes) + 1e-8
    per_sample_pct = 100.0 * per_sample_mae / denom
    mean_abs_pct = float(np.mean(per_sample_pct))
    max_abs_pct = float(np.max(per_sample_pct))
    worst = int(np.argmax(per_sample_pct))
    rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))

    detail_lines = []
    y_train = get_array(context, "y_train")
    if y_train is not None and _squeezed(y_train.shape[1:]) == _squeezed(field_shape):
        ref = y_train
    else:
        if y_train is not None:
            detail_lines.append(
                f"y_train per-sample shape {y_train.shape[1:]} does not match "
                f"the field shape {field_shape}; the mean-field baseline is "
                "derived from y_test instead"
            )
        ref = y_true
    base = ref.reshape(-1, *field_shape).mean(axis=0)
    rmse_base = float(np.sqrt(np.mean((y_true - base) ** 2)))
    skill = 1.0 - rmse / rmse_base if rmse_base > 0 else float("nan")

    warn_pct = float(cfg.get("warn_pct", 5.0))
    fail_pct = float(cfg.get("fail_pct", 20.0))
    max_rmse = cfg.get("max_rmse")

    metrics = {
        "field_rmse": rmse,
        "mean_abs_percent_error": mean_abs_pct,
        "max_abs_percent_error": max_abs_pct,
        "worst_sample_index": float(worst),
        "skill_vs_mean_field": skill,
        "warn_pct": warn_pct,
        "fail_pct": fail_pct,
    }

    figures = []
    plt = want_figures(context)
    if plt is not None and all_finite:
        # Skipped for non-finite predictions so the FAIL verdict below is
        # reached instead of matplotlib crashing on NaN/inf data.
        figures = _field_figures(plt, y_true[worst], y_hat[worst])

    if not all_finite:
        status, summary = Status.FAIL, "field predictions contain NaN or inf"
    elif max_rmse is not None and rmse > float(max_rmse):
        status = Status.FAIL
        summary = f"field RMSE {rmse:.4g} exceeds budget {float(max_rmse):.4g}"
    elif np.isfinite(skill) and skill <= 0:
        status = Status.FAIL
        summary = "no skill: field error is no better than the mean field"
    elif mean_abs_pct > fail_pct:
        status = Status.FAIL
        summary = f"mean field error {mean_abs_pct:.1f}% exceeds {fail_pct:.0f}%"
    elif mean_abs_pct > warn_pct:
        status = Status.WARN
        summary = f"mean field error {mean_abs_pct:.1f}%"
    else:
        status = Status.PASS
        summary = f"mean field error {mean_abs_pct:.1f}%, skill {skill:.2f}"

    return CheckResult(
        name=name,
        status=status,
        summary=summary,
        metrics=metrics,
        details="\n".join(detail_lines),
        figures=figures,
    )


field.check_name = "error.field"


def _field_figures(plt, true_field: np.ndarray, pred_field: np.ndarray) -> list:
    """Build the true / predicted / percent-error panel for one field sample."""
    true_field = np.asarray(true_field)
    pred_field = np.asarray(pred_field)
    pct = 100.0 * (pred_field - true_field) / (np.abs(true_field).mean() + 1e-8)

    if true_field.ndim == 2:
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
        panels = [
            (axs[0], true_field, "true", "viridis"),
            (axs[1], pred_field, "predicted", "viridis"),
            (axs[2], pct, "percent error", "coolwarm"),
        ]
        for ax, data, title, cmap in panels:
            im = ax.imshow(data, aspect="auto", cmap=cmap)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("error.field: worst sample")
        fig.tight_layout()
        return [fig]

    grid = np.arange(true_field.size)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    axs[0].plot(grid, true_field.ravel(), color="#222222", lw=1, label="true")
    axs[0].plot(grid, pred_field.ravel(), color="C0", lw=1, ls="--", label="predicted")
    axs[0].set_title("true vs predicted")
    axs[0].legend()
    axs[1].plot(grid, pct.ravel(), color="#cf222e", lw=1)
    axs[1].set_title("percent error")
    fig.suptitle("error.field: worst sample")
    fig.tight_layout()
    return [fig]
