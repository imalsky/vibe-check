"""Distribution checks: training-domain coverage and split drift.

A surrogate is only trustworthy inside the region it was trained on. These
checks compare the train / validation / test input distributions and flag test
points that sit outside the training domain, where the model is extrapolating
rather than interpolating.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import as_2d, get_array, skip, want_figures
from ..core import CheckResult, Status


def coverage(**context: Any) -> CheckResult:
    """Flag test inputs that fall outside the training domain (extrapolation).

    For each input feature the check takes the ``[min, max]`` envelope of the
    training data and counts the fraction of test points that fall outside it.
    A point outside the envelope on any feature is extrapolated, not
    interpolated, and the surrogate has no training support there.

    Verdict is on the fraction of test points that are out of domain:
    above ``fail_frac`` (default 0.10) -> FAIL, above ``warn_frac`` (default
    0.01) -> WARN, otherwise PASS. Both thresholds and an optional ``margin``
    (fraction of each feature range added to the envelope, default 0) live under
    ``metadata['coverage']``. Needs ``X_train`` and ``X_test``; SKIPs otherwise.
    Set ``metadata['make_figures'] = True`` to attach a per-feature bar chart.
    """
    name = "distribution.coverage"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("coverage") or {}
    warn_frac = float(cfg.get("warn_frac", 0.01))
    fail_frac = float(cfg.get("fail_frac", 0.10))
    margin = float(cfg.get("margin", 0.0))

    x_train = get_array(context, "X_train")
    x_test = get_array(context, "X_test")
    if x_train is None or x_test is None:
        return skip(name, "need both X_train and X_test")
    x_train = as_2d(x_train)
    x_test = as_2d(x_test)
    if x_train.shape[1] != x_test.shape[1]:
        return skip(name, "X_train and X_test have differing feature dimensions")

    lo = x_train.min(axis=0)
    hi = x_train.max(axis=0)
    span = hi - lo
    lo = lo - margin * span
    hi = hi + margin * span

    out = (x_test < lo) | (x_test > hi)
    per_feature_frac = out.mean(axis=0)
    frac_out = float(out.any(axis=1).mean())

    safe_span = np.where(span < 1e-12, 1.0, span)
    excursion = np.maximum(lo - x_test, x_test - hi) / safe_span
    max_excursion = float(np.maximum(excursion, 0.0).max()) if excursion.size else 0.0

    metrics = {
        "frac_test_out_of_domain": frac_out,
        "n_features_with_extrapolation": float(int((per_feature_frac > 0).sum())),
        "max_feature_out_of_domain_frac": (
            float(per_feature_frac.max()) if per_feature_frac.size else 0.0
        ),
        "max_excursion_in_feature_ranges": max_excursion,
    }

    figures = []
    plt = want_figures(context)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(per_feature_frac.size), per_feature_frac, color="#cf6a1a")
        ax.set_xlabel("input feature")
        ax.set_ylabel("fraction out of domain")
        ax.set_title("distribution.coverage: per-feature extrapolation")
        fig.tight_layout()
        figures.append(fig)

    if frac_out > fail_frac:
        status = Status.FAIL
        summary = f"{frac_out:.1%} of test points are outside the training domain"
    elif frac_out > warn_frac:
        status = Status.WARN
        summary = f"{frac_out:.1%} of test points are outside the training domain"
    else:
        status = Status.PASS
        summary = "test points are within the training domain"

    return CheckResult(
        name=name, status=status, summary=summary, metrics=metrics, figures=figures
    )


coverage.check_name = "distribution.coverage"


def drift(**context: Any) -> CheckResult:
    """Compare train/val/test marginals (histograms, Q-Q, KS distance)."""
    raise NotImplementedError


drift.check_name = "distribution.drift"
