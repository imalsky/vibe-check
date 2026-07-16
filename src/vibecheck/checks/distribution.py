"""Distribution checks: training-domain coverage and split drift.

A surrogate is only trustworthy inside the region it was trained on. These
checks compare the train / validation / test input distributions and flag test
points that sit outside the training domain, where the model is extrapolating
rather than interpolating.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import as_2d, get_array, ks_distance, skip, want_figures
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
    """Compare train/val/test marginals with the Kolmogorov-Smirnov distance.

    For each input feature the check computes the two-sample KS distance
    between the training marginal and each held-out marginal (validation and
    test). The KS distance is in ``[0, 1]``; larger means the distributions
    differ more. The verdict is on the worst feature across comparisons: above
    ``fail_ks`` (default 0.2) -> FAIL, above ``warn_ks`` (default 0.1) -> WARN,
    otherwise PASS. Thresholds live under ``metadata['drift']``.

    A drifted marginal does not by itself mean the surrogate is wrong, but it
    tells you the held-out data is not sampled like the training data, which is
    often why coverage and error checks then fail. Needs ``X_train`` and at
    least one of ``X_val`` / ``X_test``; SKIPs otherwise. Set
    ``metadata['make_figures'] = True`` for a per-feature KS bar chart.
    """
    name = "distribution.drift"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("drift") or {}
    warn_ks = float(cfg.get("warn_ks", 0.1))
    fail_ks = float(cfg.get("fail_ks", 0.2))

    x_train = get_array(context, "X_train")
    if x_train is None:
        return skip(name, "no X_train")
    x_train = as_2d(x_train)
    n_feat = x_train.shape[1]

    others = {
        key: as_2d(get_array(context, key))
        for key in ("X_val", "X_test")
        if context.get(key) is not None
    }
    comparisons = {
        label: np.array([ks_distance(x_train[:, j], x[:, j]) for j in range(n_feat)])
        for label, x in others.items()
        if x.shape[1] == n_feat
    }
    if not comparisons:
        return skip(name, "need X_val or X_test with matching feature dimensions")

    worst = 0.0
    worst_label = ""
    worst_feature = -1
    for label, ks in comparisons.items():
        j = int(np.nanargmax(ks))
        if ks[j] > worst:
            worst, worst_label, worst_feature = float(ks[j]), label, j

    all_ks = np.concatenate(list(comparisons.values()))
    metrics = {
        "max_ks_distance": worst,
        "worst_feature_index": float(worst_feature),
        "mean_ks_distance": float(np.nanmean(all_ks)),
        "warn_ks": warn_ks,
        "fail_ks": fail_ks,
    }
    details = "\n".join(
        f"- train vs {label}: max KS {np.nanmax(ks):.3f} at feature {int(np.nanargmax(ks))}"
        for label, ks in comparisons.items()
    )

    figures = []
    plt = want_figures(context)
    if plt is not None:
        label = "X_test" if "X_test" in comparisons else next(iter(comparisons))
        ks = comparisons[label]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(n_feat), ks, color="#1a6acf")
        ax.axhline(warn_ks, ls="--", lw=1, color="#9a6700")
        ax.axhline(fail_ks, ls="--", lw=1, color="#cf222e")
        ax.set_xlabel("input feature")
        ax.set_ylabel("KS distance")
        ax.set_title(f"distribution.drift: train vs {label}")
        fig.tight_layout()
        figures.append(fig)

    if worst > fail_ks:
        status = Status.FAIL
        summary = (
            f"strong marginal drift (max KS {worst:.2f}) between train and "
            f"{worst_label} at feature {worst_feature}"
        )
    elif worst > warn_ks:
        status = Status.WARN
        summary = f"moderate marginal drift (max KS {worst:.2f}) vs {worst_label}"
    else:
        status = Status.PASS
        summary = f"train and held-out marginals are close (max KS {worst:.2f})"

    return CheckResult(
        name=name,
        status=status,
        summary=summary,
        metrics=metrics,
        details=details,
        figures=figures,
    )


drift.check_name = "distribution.drift"
