# Tutorial: from a trained surrogate to a report

This walks through validating a surrogate with `vibe-check`, end to end. It
assumes you already have a trained model; the package does not train anything,
it checks what you bring.

## Install

```
pip install -e ".[viz]"
```

The core depends only on numpy. The `viz` extra adds matplotlib for figures.
Framework-specific code (torch, onnx) is never imported unless a check needs it.

## The one thing vibe-check needs

Whatever your model is, reduce it to a `predict` callable that maps a batch of
inputs to a batch of outputs:

```
predict(X) -> y_hat        # numpy in, numpy out, batched
```

Then hand it your data splits. Everything else is optional and only unlocks
more checks.

## Quickstart

This is a complete, runnable script. The "surrogate" is a linear least-squares
fit, so it needs nothing beyond numpy, but the workflow is identical for a
neural network: wrap it in `predict` and pass your splits.

```python
import numpy as np
import vibecheck as vc

rng = np.random.default_rng(0)
w_true = rng.normal(size=(4, 2))

def sample(n):
    x = rng.normal(size=(n, 4))
    y = x @ w_true + rng.normal(0.0, 0.1, size=(n, 2))
    return x, y

x_train, y_train = sample(400)
x_test, y_test = sample(200)
w_fit, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

def predict(x):
    return np.asarray(x) @ w_fit

report = vc.check(
    predict,
    X_train=x_train, y_train=y_train,
    X_test=x_test, y_test=y_test,
    metadata={
        "normalization": {"mean": x_train.mean(0), "std": x_train.std(0)},
        "speed": {"min_throughput": 1e4},
    },
)
print("overall:", report.summary().value)
report.to_markdown("report.md")
report.to_html("report.html")
```

Running it prints something like:

```
overall: warn
  leakage.normalization: pass
  leakage.split_overlap: pass
  distribution.coverage: pass
  distribution.drift: warn
  error.pointwise: pass
  error.field: skip
  calibration.coverage: skip
  constraints.physical: skip
  export.roundtrip: skip
  speed.inference: pass
```

Every registered check ran. Some passed, some had nothing to work with and
returned `skip` (a first-class outcome, never a silent pass), and one marginal
drifted enough to `warn`. Open `report.html` to see the same thing with figures.

## Reading the report

Each check returns a `CheckResult` with a `status`, a one-line `summary`, and
machine-readable `metrics`. The five statuses:

- `PASS` - the check ran and the surrogate met it.
- `WARN` - worth a look, not necessarily wrong.
- `FAIL` - a real problem the check is confident about.
- `SKIP` - the check could not run (its inputs were not provided). Not a pass.
- `ERROR` - the check itself raised; treated as distinct from FAIL.

`report.summary()` returns the worst status across all checks, which is what you
gate CI on:

```python
if report.summary() is vc.Status.FAIL:
    raise SystemExit("surrogate failed validation")
```

## Turning on more checks

Checks that need more than the raw arrays read from the optional `metadata`
dict. A check that does not find what it needs returns `SKIP`, so you enable a
check simply by providing its inputs:

```python
metadata = {
    # leakage.normalization: the scaler stats you actually applied.
    "normalization": {"mean": x_train.mean(0), "std": x_train.std(0)},

    # calibration.coverage: a predictive std (or return (mean, std) from predict).
    "predicted_std": per_sample_std,

    # constraints.physical: declare what the outputs must obey.
    "constraints": [
        {"type": "positivity"},
        {"type": "sum", "value": 1.0, "axis": -1, "tol": 1e-3},
    ],

    # export.roundtrip: a second callable for the exported model.
    "exported_predict": onnx_session_wrapped_as_a_callable,

    # speed.inference: state the budget so "fast enough" is measured.
    "speed": {"min_throughput": 1e4},

    # attach matplotlib figures (off by default so checks stay cheap).
    "make_figures": True,
}
```

The full set of metadata keys, per check, is tabulated in
[`VALIDATION_CONTRACT.md`](VALIDATION_CONTRACT.md).

## Tuning and turning off checks

Each check reads its thresholds from `metadata`, so you can loosen or tighten a
check without code changes, for example `metadata["drift"] = {"warn_ks": 0.15,
"fail_ks": 0.3}`. To skip a check entirely, simply do not provide its inputs; it
returns `SKIP` and stays out of the way.

## Next steps

The [`examples/`](../examples) directory has three worked surrogates (Robertson
kinetics, the Lorenz system, and an FNO on Burgers) with committed reports you
can use as templates for your own model.
