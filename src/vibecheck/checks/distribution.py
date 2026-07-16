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
    interpolated, and the surrogate has no training support there. Assumes the
    train and test splits share the same feature columns and that rows are
    comparable samples.

    Verdict is on the fraction of test points that are out of domain: above
    the fail threshold -> FAIL (a substantial share of test points has no
    training support), above the warn threshold -> WARN (a small but real
    share is extrapolated), otherwise PASS. By default the warn threshold
    adapts upward for small samples, since even iid draws from the training
    distribution land outside the observed envelope at a rate of about
    ``2 * n_features / (n_train + 1)``, so healthy data rarely warns; explicit
    ``warn_frac`` / ``fail_frac`` values under ``metadata['coverage']``
    override this and are used verbatim. An optional ``margin`` (fraction of
    each feature range added to the envelope, default 0) lives there too. The
    effective thresholds are echoed in the metrics. Needs ``X_train`` and
    ``X_test``; SKIPs otherwise. Set ``metadata['make_figures'] = True`` to
    attach a per-feature bar chart.
    """
    name = "distribution.coverage"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("coverage") or {}
    user_warn = cfg.get("warn_frac")
    user_fail = cfg.get("fail_frac")
    margin = float(cfg.get("margin", 0.0))

    x_train = get_array(context, "X_train")
    x_test = get_array(context, "X_test")
    if x_train is None or x_test is None:
        return skip(name, "need both X_train and X_test")
    x_train = as_2d(x_train)
    x_test = as_2d(x_test)
    if x_train.shape[1] != x_test.shape[1]:
        return skip(name, "X_train and X_test have differing feature dimensions")

    # Adaptive defaults: iid test data lands outside the observed train
    # envelope at roughly this rate, so warn only well above it. Explicit
    # metadata thresholds are used verbatim.
    n_train, n_features = x_train.shape
    m = x_test.shape[0]
    expected = min(1.0, 2.0 * n_features / (n_train + 1))
    if user_warn is not None:
        warn_frac = float(user_warn)
    else:
        spread = np.sqrt(expected * (1.0 - expected) / max(m, 1))
        warn_frac = max(0.01, expected + 3.0 * float(spread))
    if user_fail is not None:
        fail_frac = float(user_fail)
    else:
        fail_frac = max(0.10, 2.0 * warn_frac)

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
        "warn_frac": float(warn_frac),
        "fail_frac": float(fail_frac),
    }

    figures = []
    plt = want_figures(context)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(per_feature_frac.size), per_feature_frac, color="C1")
        ax.set_ylim(bottom=0)
        if per_feature_frac.size and per_feature_frac.max() == 0:
            ax.text(
                0.5,
                0.5,
                "no out-of-domain points detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        if per_feature_frac.size <= 30:
            ax.set_xticks(np.arange(per_feature_frac.size))
        ax.set_xlabel("input feature")
        ax.set_ylabel("fraction out of domain")
        ax.set_title("distribution.coverage: per-feature extrapolation")
        fig.tight_layout()
        figures.append(fig)

    if frac_out > fail_frac:
        status = Status.FAIL
        summary = (
            f"{frac_out:.1%} of test points are outside the training domain "
            f"(above fail threshold {fail_frac:.1%})"
        )
    elif frac_out > warn_frac:
        status = Status.WARN
        summary = (
            f"{frac_out:.1%} of test points outside the training domain "
            f"(above warn threshold {warn_frac:.1%})"
        )
    else:
        status = Status.PASS
        summary = (
            f"{frac_out:.1%} of test points outside the training domain "
            f"(within tolerance)"
        )

    return CheckResult(
        name=name, status=status, summary=summary, metrics=metrics, figures=figures
    )


coverage.check_name = "distribution.coverage"


def drift(**context: Any) -> CheckResult:
    """Compare train/val/test marginals with the Kolmogorov-Smirnov distance.

    For each input feature the check computes the two-sample KS distance
    between the training marginal and each held-out marginal (validation and
    test). The KS distance is in ``[0, 1]``; larger means the distributions
    differ more. Assumes each column is the same feature across splits. The
    verdict is on the worst feature across comparisons: above the fail
    threshold -> FAIL (the held-out split is clearly not sampled like the
    training data), above the warn threshold -> WARN (moderate drift worth
    inspecting), otherwise PASS. By default the thresholds adapt upward for
    small samples, using the two-sample KS critical scale
    ``1.7 * sqrt((n + m) / (n * m))`` per comparison, so that data drawn iid
    from the training distribution rarely warns; explicit ``warn_ks`` /
    ``fail_ks`` values under ``metadata['drift']`` override this and are used
    verbatim. The effective thresholds for the worst comparison are echoed in
    the metrics.

    A drifted marginal does not by itself mean the surrogate is wrong, but it
    tells you the held-out data is not sampled like the training data, which is
    often why coverage and error checks then fail. Needs ``X_train`` and at
    least one of ``X_val`` / ``X_test``; SKIPs otherwise. Set
    ``metadata['make_figures'] = True`` for a per-feature KS bar chart plus a
    histogram and quantile-quantile view of the worst-drifting feature.
    """
    name = "distribution.drift"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("drift") or {}
    user_warn = cfg.get("warn_ks")
    user_fail = cfg.get("fail_ks")

    x_train = get_array(context, "X_train")
    if x_train is None:
        return skip(name, "no X_train")
    x_train = as_2d(x_train)
    n_train, n_feat = x_train.shape

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

    # Per-comparison effective thresholds: adaptive by default so that iid
    # samples rarely exceed them, verbatim when set in metadata.
    thresholds = {}
    for label in comparisons:
        m = others[label].shape[0]
        crit = 1.7 * float(np.sqrt((n_train + m) / (n_train * m)))
        warn_eff = float(user_warn) if user_warn is not None else max(0.1, crit)
        fail_eff = float(user_fail) if user_fail is not None else max(0.2, 2.0 * crit)
        thresholds[label] = (warn_eff, fail_eff)

    ranked = []
    for label, ks in comparisons.items():
        j = int(np.nanargmax(ks))
        value = float(ks[j])
        warn_eff, fail_eff = thresholds[label]
        if value > fail_eff:
            rank = 2
        elif value > warn_eff:
            rank = 1
        else:
            rank = 0
        ranked.append((rank, value, label, j))
    rank, worst, worst_label, worst_feature = max(ranked)
    warn_ks, fail_ks = thresholds[worst_label]

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
        fig_label = "X_test" if "X_test" in comparisons else next(iter(comparisons))
        ks = comparisons[fig_label]
        warn_line, fail_line = thresholds[fig_label]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(n_feat), ks, color="C0")
        ax.axhline(
            warn_line, ls="--", lw=1, color="#9a6700", label=f"warn {warn_line:g}"
        )
        ax.axhline(
            fail_line, ls="--", lw=1, color="#cf222e", label=f"fail {fail_line:g}"
        )
        if n_feat <= 30:
            ax.set_xticks(np.arange(n_feat))
        # Headroom above the fail line so the legend does not sit on it.
        ax.set_ylim(0, max(fail_line * 1.35, float(np.nanmax(ks)) * 1.2))
        ax.set_xlabel("input feature")
        ax.set_ylabel("KS distance")
        ax.set_title(f"distribution.drift: train vs {fig_label}")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        fig.tight_layout()
        figures.append(fig)

        # Histogram (doubles as a density plot) and quantile-quantile view of
        # the worst-drifting feature of the worst comparison.
        train_col = x_train[:, worst_feature]
        other_col = others[worst_label][:, worst_feature]
        fig2, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(8, 3))
        bins = np.histogram_bin_edges(
            np.concatenate([train_col, other_col]), bins=30
        )
        ax_hist.hist(
            train_col, bins=bins, density=True, alpha=0.5, color="C0",
            label="train",
        )
        ax_hist.hist(
            other_col, bins=bins, density=True, alpha=0.5, color="C1",
            label=worst_label,
        )
        ax_hist.set_xlabel(f"feature {worst_feature}")
        ax_hist.set_ylabel("density")
        ax_hist.legend(frameon=False, fontsize=8)
        qs = np.linspace(0.01, 0.99, 99)
        q_train = np.quantile(train_col, qs)
        q_other = np.quantile(other_col, qs)
        ax_qq.plot(q_train, q_other, "o", ms=3, color="C0")
        lims = [
            min(float(q_train.min()), float(q_other.min())),
            max(float(q_train.max()), float(q_other.max())),
        ]
        ax_qq.plot(lims, lims, ls="--", lw=1, color="#57606a", label="y = x")
        ax_qq.set_xlabel("train quantiles")
        ax_qq.set_ylabel(f"{worst_label} quantiles")
        ax_qq.legend(loc="upper left", frameon=False, fontsize=8)
        fig2.suptitle(
            f"distribution.drift: feature {worst_feature} (train vs {worst_label})",
            fontsize=10,
        )
        fig2.tight_layout(rect=(0, 0, 1, 0.93))
        figures.append(fig2)

    if rank == 2:
        status = Status.FAIL
        summary = (
            f"strong marginal drift (max KS {worst:.2f}) between train and "
            f"{worst_label} at feature {worst_feature}"
        )
    elif rank == 1:
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
