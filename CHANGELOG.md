# Changelog

Notable changes to vibe-check are recorded here. The project follows semantic
versioning; while the major version is 0 the public API may still change.

## 0.1.0 - unreleased

First functional release. Every check in the validation contract is implemented,
registered, tested, and documented.

### Added

- Core contract: `Status`, `CheckResult`, and `Report` (with `to_markdown` and a
  self-contained `to_html` that embeds figures as base64 PNG), plus the `check`
  orchestrator that runs the registered diagnostics and wraps failures.
- Ten reliability checks: `leakage.normalization`, `leakage.split_overlap`,
  `distribution.coverage`, `distribution.drift`, `error.pointwise`,
  `error.field`, `calibration.coverage`, `constraints.physical`,
  `export.roundtrip`, and `speed.inference`. Each returns a `SKIP` when its
  inputs are absent rather than passing silently, and reads its thresholds from
  `metadata`.
- Three worked examples with committed reports: Robertson stiff kinetics
  (state-to-state MLP), the Lorenz system (ordered-output trajectory), and a
  Fourier Neural Operator on Burgers (spatial field).
- Documentation: the validation contract with a metadata-conventions table, an
  end-to-end tutorial, and a case study on normalization leakage with a runnable
  demonstration.
- Continuous integration across Python 3.10, 3.11, and 3.12 (ruff and pytest).
