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
    assessed from training data alone). The mean discrepancy is scaled by the
    reference standard deviation (so centered, near-zero-mean features do not
    inflate the distance) and the std discrepancy is relative to the reference
    std. The match tolerance is tunable via
    ``metadata['normalization']['rtol']`` (default ``1e-3``): the distance
    under which provided statistics count as a match.
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

    def mean_dist(a: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> float:
        # The mean gap is scaled by the reference std, not the reference mean:
        # centered features have a near-zero mean, and dividing by it would
        # blow up the distance for perfectly correct statistics.
        return float(np.mean(np.abs(a - ref_mean) / (ref_std + 1e-12)))

    train_mean, train_std = x_train.mean(axis=0), x_train.std(axis=0)
    full_mean, full_std = x_all.mean(axis=0), x_all.std(axis=0)
    d_train = 0.5 * (
        mean_dist(mean_provided, train_mean, train_std) + rel(std_provided, train_std)
    )
    d_full = 0.5 * (
        mean_dist(mean_provided, full_mean, full_std) + rel(std_provided, full_std)
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
    """Detect duplicate / near-duplicate samples shared across data splits.

    Rows are standardized by the training statistics and, for every pair of
    available splits, each row's nearest neighbor in the other split is found.
    A distance of zero is an exact duplicate; a small non-zero distance is a
    near-duplicate. The verdict:

    - any exact duplicate across splits -> FAIL (the held-out set is not truly
      held out);
    - only near-duplicates -> WARN (leakage-like, worth a look);
    - neither -> PASS.

    Needs at least two of ``X_train`` / ``X_val`` / ``X_test``; SKIPs otherwise.
    The near-duplicate threshold on the per-feature standardized distance is
    ``metadata['split_overlap']['atol']`` (default ``1e-3``). Splits larger than
    ``metadata['split_overlap']['max_rows']`` (default 3000) are subsampled with
    a fixed seed and the subsampling is noted in the report.
    """
    name = "leakage.split_overlap"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("split_overlap") or {}
    atol = float(cfg.get("atol", 1e-3))
    cap = int(cfg.get("max_rows", 3000))

    raw = {
        key: as_2d(get_array(context, key))
        for key in ("X_train", "X_val", "X_test")
        if context.get(key) is not None
    }
    if len(raw) < 2:
        return skip(name, "need at least two of X_train / X_val / X_test")

    ref = raw.get("X_train")
    if ref is None:
        ref = np.concatenate(list(raw.values()), axis=0)
    n_feat = ref.shape[1]
    if any(s.shape[1] != n_feat for s in raw.values()):
        return skip(name, "splits have differing feature dimensions")

    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    rng = np.random.default_rng(0)

    def prep(x: np.ndarray) -> np.ndarray:
        if x.shape[0] > cap:
            idx = rng.choice(x.shape[0], size=cap, replace=False)
            x = x[idx]
        return (x - mu) / sd

    subsampled = any(v.shape[0] > cap for v in raw.values())
    std_splits = {k: prep(v) for k, v in raw.items()}

    keys = list(std_splits)
    total_exact = 0
    total_near = 0
    overall_min = float("inf")
    detail_lines = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            n_exact, n_near, min_d = _cross_overlap(
                std_splits[a_key], std_splits[b_key], atol
            )
            total_exact += n_exact
            total_near += n_near
            if min_d == min_d:  # not NaN
                overall_min = min(overall_min, min_d)
            detail_lines.append(
                f"- {a_key} vs {b_key}: {n_exact} exact, {n_near} near-duplicate rows"
            )
    if subsampled:
        detail_lines.append(f"- note: splits over {cap} rows were subsampled")

    metrics = {
        "exact_duplicate_rows": float(total_exact),
        "near_duplicate_rows": float(total_near),
        "min_standardized_distance": (
            overall_min if overall_min != float("inf") else float("nan")
        ),
        "atol": atol,
    }
    details = "\n".join(detail_lines)

    if total_exact > 0:
        return CheckResult(
            name=name,
            status=Status.FAIL,
            summary=(
                f"{total_exact} row(s) appear in more than one split: "
                "the held-out set is not truly held out"
            ),
            metrics=metrics,
            details=details,
        )
    if total_near > 0:
        return CheckResult(
            name=name,
            status=Status.WARN,
            summary=(
                f"{total_near} near-duplicate row(s) across splits "
                f"(standardized distance < {atol})"
            ),
            metrics=metrics,
            details=details,
        )
    return CheckResult(
        name=name,
        status=Status.PASS,
        summary="no duplicate or near-duplicate rows detected across splits",
        metrics=metrics,
        details=details,
    )


split_overlap.check_name = "leakage.split_overlap"


def _cross_overlap(a: np.ndarray, b: np.ndarray, atol: float) -> tuple[int, int, float]:
    """Nearest-neighbor overlap of the rows of ``a`` against ``b``.

    Both arrays are already standardized. Returns the number of rows of ``a``
    that are exact duplicates of some row of ``b``, the number that are
    near-duplicates (per-feature standardized distance below ``atol``), and the
    smallest such distance. Distances are computed in memory-bounded chunks.
    """
    na, nb = a.shape[0], b.shape[0]
    n_feat = a.shape[1]
    if na == 0 or nb == 0:
        return 0, 0, float("nan")
    budget = 5_000_000
    chunk = max(1, budget // max(1, nb * n_feat))
    min_d = np.empty(na)
    for start in range(0, na, chunk):
        block = a[start : start + chunk]
        dist = np.sqrt(((block[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        min_d[start : start + chunk] = dist.min(axis=1)
    d_norm = min_d / np.sqrt(n_feat)
    n_exact = int(np.sum(d_norm <= 1e-9))
    n_near = int(np.sum((d_norm > 1e-9) & (d_norm < atol)))
    return n_exact, n_near, float(d_norm.min())
