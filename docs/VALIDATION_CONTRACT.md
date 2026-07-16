# Validation contract

This document defines the shared interface that every `vibe-check` diagnostic
follows. The core types below (`Status`, `CheckResult`, `Report`) are
implemented and stable in shape as of 0.1.0. The set of checks and their
default thresholds are still expected to change as the community weighs in, so
treat those as tunable, not frozen.

## Design goals

1. A user brings a trained surrogate and their data splits. They get back a
   report that a non-author can read.
2. Every check is independent, cheap by default, and honest about what it did
   and did not test.
3. No check should ever silently pass. "Not applicable" and "not run" are
   first-class outcomes, distinct from "pass".

## The surrogate, as seen by vibe-check

`vibe-check` does not depend on any ML framework. A surrogate is reduced to a
callable:

```
predict: Callable[[ArrayLike], ArrayLike]   # X -> y_hat, batched
```

plus, optionally, the arrays used to build and evaluate it:

```
X_train, y_train, X_val, y_val, X_test, y_test : ArrayLike
```

and optional metadata describing channels, units, and constraints.

## CheckResult

Every check returns a `CheckResult`:

- `name: str`: stable identifier, e.g. `"leakage.normalization"`.
- `status: Status`: one of `PASS`, `WARN`, `FAIL`, `SKIP`, `ERROR`.
- `summary: str`: one line a non-author can understand.
- `metrics: dict[str, float]`: machine-readable numbers behind the verdict.
- `figures: list[Figure]`: optional matplotlib figures (only if `viz` used).
- `details: str`: optional longer explanation, markdown allowed.

## Report

A `Report` aggregates `CheckResult`s and renders them:

- `report.to_markdown(path)`: text report, no heavy deps.
- `report.to_html(path)`: self-contained HTML with embedded figures (`viz`).
- `report.summary()`: the worst status across checks, for CI gating.

Both renderers accept an optional `title` keyword so a report can name the
surrogate it describes. `print(report)` gives a terse per-check listing.

## The core checks

All ten are implemented, registered, and tested.

| id                        | detects                                                        |
|---------------------------|---------------------------------------------------------------|
| `leakage.normalization`   | scaler/stats fit on test or full data instead of train only   |
| `leakage.split_overlap`   | duplicate or near-duplicate rows across train/test            |
| `distribution.coverage`   | test inputs outside the training domain (extrapolation)       |
| `distribution.drift`      | train vs val vs test marginal drift (per-feature KS distance; histogram and Q-Q figures) |
| `error.pointwise`         | predicted-vs-true, residuals, per-channel error tables        |
| `error.field`             | field-level true / predicted / percent-error maps             |
| `calibration.coverage`    | stated uncertainty vs empirical coverage (Gaussian intervals) |
| `constraints.physical`    | conservation, positivity, monotonicity, symmetry, bounds      |
| `export.roundtrip`        | exported model output matches in-memory model                 |
| `speed.inference`         | throughput / latency vs a user-set budget                     |

Each check documents: what it assumes, what a WARN vs FAIL means, and how to
turn it off or tune its thresholds.

## Metadata conventions

Checks that need more than the arrays read from a shared, optional `metadata`
dict. Every key is optional; a check that does not find what it needs returns
`SKIP`. The keys in use today:

| key                          | used by                 | meaning                                                           |
|------------------------------|-------------------------|-------------------------------------------------------------------|
| `normalization`              | `leakage.normalization` | `{"mean", "std"}` (optional `rtol`): the scaler stats you applied |
| `split_overlap`              | `leakage.split_overlap` | `{"atol", "max_rows"}`: near-duplicate threshold and subsample cap|
| `coverage`                   | `distribution.coverage` | `{"warn_frac", "fail_frac", "margin"}`                            |
| `drift`                      | `distribution.drift`    | `{"warn_ks", "fail_ks"}`                                          |
| `error`                      | `error.pointwise`       | `{"max_rmse", "warn_skill"}`                                      |
| `error_field`                | `error.field`           | `{"warn_pct", "fail_pct", "max_rmse", "min_field_size", "is_field"}` |
| `predicted_std`              | `calibration.coverage`  | array aligned with `y_test`, or a scalar (or return `(mean, std)`)|
| `calibration`                | `calibration.coverage`  | `{"warn_tol", "fail_tol"}`                                        |
| `constraints`                | `constraints.physical`  | list of constraint specs (see that check's docstring)             |
| `constraints_config`         | `constraints.physical`  | `{"fail_frac"}`                                                   |
| `exported_predict`           | `export.roundtrip`      | a callable `X -> y_hat` for the exported model                    |
| `export`                     | `export.roundtrip`      | `{"atol", "rtol", "warn_frac"}`                                   |
| `speed`                      | `speed.inference`       | `{"min_throughput", "max_latency_s", "warmup", "repeats"}`        |
| `make_figures`               | all viz-capable checks  | set `True` to attach matplotlib figures (off by default)          |
