# vibe-check

Reliability checks for machine-learning surrogates in scientific software.

`vibe-check` is a lightweight Python package. It tests whether an ML surrogate
(an emulator, neural operator, or learned solver component) can actually be
trusted before it goes into a larger scientific software stack. Aggregate
accuracy hides failures: a model with low average error can still extrapolate
silently, leak information during preprocessing, or violate known physical
constraints. `vibe-check` turns these common failure modes into clear,
attachable validation reports.

This work is supported by a 2026 URSSI Fellowship (Code for Science & Society).

## Status

Version 0.1.0. All ten checks in the validation contract are implemented,
tested, and documented, with three worked examples. The public API is close to
stable, but it may still change before 1.0. See `CHANGELOG.md` and
`ROADMAP.md`.

## What it checks

- **Leakage**: a scaler fit on the wrong data split, or overlap between train
  and test.
- **Distribution coverage**: drift between train, validation, and test data;
  where the model is being asked to extrapolate.
- **Error**: predicted vs. true values, residuals, per-channel and
  field-level error maps.
- **Calibration**: whether the model's stated uncertainties are honest.
- **Constraints**: conservation, positivity, monotonicity, symmetry, bounds.
- **Export**: whether the saved/exported model reproduces the in-memory one.
- **Speed**: whether inference is actually fast enough for its intended use.

## Representative examples

Three small surrogates that cover common scientific ML shapes:

- **Robertson** stiff chemical kinetics: a local state-to-state emulator.
- **Lorenz** system: an ordered-output trajectory emulator.
- **Fourier Neural Operator** (Burgers): a spatial-field emulator.

## Install (development)

```
pip install -e ".[dev,viz]"
```

## Quick start

```python
import vibecheck as vc

report = vc.check(
    predict=model.predict,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
)
print(report)
report.to_markdown("report.md")
```

`predict` takes a batch of inputs and returns a batch of outputs, numpy in and
numpy out. A check whose inputs you did not provide reports SKIP instead of
silently passing. `docs/TUTORIAL.md` lists the metadata keys that turn on more
checks.

## Documentation

- `docs/TUTORIAL.md` - from a trained surrogate to a report, end to end.
- `docs/VALIDATION_CONTRACT.md` - the interface every check follows and the
  `metadata` keys each one reads.
- `examples/` - three worked surrogates with committed reports.

## License

MIT. See `LICENSE`.
