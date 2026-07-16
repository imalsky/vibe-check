"""Leakage checks: normalization hygiene and train/test overlap.

Data leakage is the single most common way a scientific ML result is silently
inflated (Kapoor & Narayanan 2023). Two concrete failures to detect:

- ``normalization``: the scaler / per-channel statistics were fit on the test
  set or on the full dataset instead of on training data only.
- ``split_overlap``: identical or near-identical rows appear in more than one
  split, so the test set is not actually held out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import as_2d, get_array, skip
from ..core import CheckResult, Status


def normalization(**context: Any) -> CheckResult:
    """Detect normalization statistics that were not fit on train-only data.

    The user passes the per-feature statistics they actually applied via
    ``metadata['normalization'] = {'mean': ..., 'std': ...}``. The check
    recomputes the train-only statistics and the full-data statistics (train
    plus any validation/test split) and asks which the provided statistics
    match:

    - match the train-only statistics -> PASS (correct hygiene);
    - match the full-data statistics instead -> FAIL (the scaler saw held-out
      data, the classic normalization-leakage bug);
    - match neither -> WARN (statistics of unknown provenance).

    Assumes the provided statistics are per-feature and aligned with the input
    columns. SKIPs when the statistics or the training split are missing, or
    when there is no held-out split to compare against (leakage cannot be
    assessed from training data alone). The match tolerance is tunable via
    ``metadata['normalization']['rtol']`` (default ``1e-3``): the relative
    distance under which provided statistics count as a match.
    """
    name = "leakage.normalization"
    metadata = context.get("metadata") or {}
    norm = metadata.get("normalization")
    if not isinstance(norm, dict) or "mean" not in norm or "std" not in norm:
        return skip(name, "no metadata['normalization'] with 'mean' and 'std'")

    x_train = get_array(context, "X_train")
    if x_train is None:
        return skip(name, "no X_train to recompute train-only statistics")

    held_out = [
        as_2d(get_array(context, key))
        for key in ("X_val", "X_test")
        if context.get(key) is not None
    ]
    if not held_out:
        return skip(name, "no validation/test split to compare against")

    x_train = as_2d(x_train)
    x_all = np.concatenate([x_train, *held_out], axis=0)

    mean_provided = np.asarray(norm["mean"], dtype=float).ravel()
    std_provided = np.asarray(norm["std"], dtype=float).ravel()
    n_features = x_train.shape[1]
    if mean_provided.size != n_features or std_provided.size != n_features:
        return skip(
            name,
            f"provided statistics have {mean_provided.size} entries but the "
            f"inputs have {n_features} features",
        )

    rtol = float(norm.get("rtol", 1e-3))

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b) / (np.abs(b) + 1e-12)))

    d_train = 0.5 * (
        rel(mean_provided, x_train.mean(axis=0)) + rel(std_provided, x_train.std(axis=0))
    )
    d_full = 0.5 * (
        rel(mean_provided, x_all.mean(axis=0)) + rel(std_provided, x_all.std(axis=0))
    )
    metrics = {
        "rel_distance_to_train_stats": d_train,
        "rel_distance_to_full_stats": d_full,
        "rtol": rtol,
    }

    if d_train <= rtol:
        return CheckResult(
            name=name,
            status=Status.PASS,
            summary="normalization statistics match the train-only statistics",
            metrics=metrics,
        )
    if d_full <= rtol and d_full < d_train:
        return CheckResult(
            name=name,
            status=Status.FAIL,
            summary=(
                "normalization statistics match full-data statistics, not "
                "train-only: preprocessing leaked held-out data"
            ),
            metrics=metrics,
            details=(
                "The per-feature mean/std you applied are consistent with "
                "statistics computed over train plus held-out data, not the "
                "training split alone. Refit the scaler on X_train only and "
                "apply it to the other splits."
            ),
        )
    return CheckResult(
        name=name,
        status=Status.WARN,
        summary="normalization statistics match neither train-only nor full-data statistics",
        metrics=metrics,
        details=(
            "The provided statistics could not be traced to the training split "
            "or to the full dataset. Confirm how the scaler was fit."
        ),
    )


normalization.check_name = "leakage.normalization"


def split_overlap(**context: Any) -> CheckResult:
    """Detect duplicate / near-duplicate samples shared across data splits."""
    raise NotImplementedError


split_overlap.check_name = "leakage.split_overlap"
