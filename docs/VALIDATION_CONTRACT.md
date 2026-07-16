# Validation contract (draft)

This document defines the shared interface that every `vibe-check` diagnostic
follows. It is the Month 1 deliverable ("define the validation contract"). It
is a draft and is expected to change as the checks are implemented and as the
community weighs in. Treat the shapes below as a starting point, not a frozen
API.

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

- `name: str` — stable identifier, e.g. `"leakage.normalization"`.
- `status: Status` — one of `PASS`, `WARN`, `FAIL`, `SKIP`, `ERROR`.
- `summary: str` — one line a non-author can understand.
- `metrics: dict[str, float]` — machine-readable numbers behind the verdict.
- `figures: list[Figure]` — optional matplotlib figures (only if `viz` used).
- `details: str` — optional longer explanation, markdown allowed.

## Report

A `Report` aggregates `CheckResult`s and renders them:

- `report.to_markdown(path)` — text report, no heavy deps.
- `report.to_html(path)` — self-contained HTML with embedded figures (`viz`).
- `report.summary()` — the worst status across checks, for CI gating.

## The core checks (target set)

| id                        | detects                                                        |
|---------------------------|---------------------------------------------------------------|
| `leakage.normalization`   | scaler/stats fit on test or full data instead of train only   |
| `leakage.split_overlap`   | duplicate or near-duplicate rows across train/test            |
| `distribution.coverage`   | test inputs outside the training domain (extrapolation)       |
| `distribution.drift`      | train vs val vs test marginal drift (histograms, Q-Q, KS)     |
| `error.pointwise`         | predicted-vs-true, residuals, per-channel error tables        |
| `error.field`             | field-level true / predicted / percent-error maps             |
| `calibration.coverage`    | stated uncertainty vs empirical coverage (conformal-style)    |
| `constraints.physical`    | conservation, positivity, monotonicity, symmetry, bounds      |
| `export.roundtrip`        | exported model output matches in-memory model                 |
| `speed.inference`         | throughput / latency vs a user-set budget                     |

Each check documents: what it assumes, what a WARN vs FAIL means, and how to
turn it off or tune its thresholds.
