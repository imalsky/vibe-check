# vibe-check

Reliability checks for machine-learning surrogates in scientific software.

`vibe-check` is a lightweight Python package that tests whether an ML surrogate
(emulator, neural operator, learned solver component) is actually trustworthy
before it is dropped into a larger scientific software stack. Aggregate accuracy
hides failures: a model with low average error can still extrapolate silently,
leak information during preprocessing, or violate known physical constraints.
`vibe-check` turns the common failure modes of scientific ML into clear,
attachable validation reports.

This work is supported by a 2026 URSSI Fellowship (Code for Science & Society).

## Status

Version 0.1.0: all ten checks in the validation contract are implemented,
tested, and documented, with three worked examples. The public API is close to
stable but may still change before 1.0. See `CHANGELOG.md` and `ROADMAP.md`.

## What it checks

- **Leakage** — normalization fit on the wrong split, train/test overlap.
- **Distribution coverage** — train vs. validation vs. test drift; where the
  model is being asked to extrapolate.
- **Error** — predicted-vs-true, residuals, per-channel and field-level maps.
- **Calibration** — are the model's stated uncertainties honest.
- **Constraints** — conservation, positivity, monotonicity, symmetry, bounds.
- **Export** — does the saved/exported model reproduce the in-memory one.
- **Speed** — is inference actually fast enough for the intended use.

## Representative examples

Three small surrogates that cover common scientific ML shapes:

- **Robertson** stiff chemical kinetics — a local state-to-state emulator.
- **Lorenz** system — an ordered-output trajectory emulator.
- **Fourier Neural Operator** (Burgers / Darcy) — a spatial-field emulator.

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
report.to_markdown("report.md")
```

## Documentation

- `docs/TUTORIAL.md` - from a trained surrogate to a report, end to end.
- `docs/VALIDATION_CONTRACT.md` - the interface every check follows and the
  `metadata` keys each one reads.
- `examples/` - three worked surrogates with committed reports.

## License

MIT. See `LICENSE`.
