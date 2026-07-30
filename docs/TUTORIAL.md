# Tutorial: from a trained surrogate to a report

This walks through validating a surrogate with `vibe-check`, end to end. It
assumes you already have a trained model. The package does not train anything;
it checks what you bring.

## Install

From a clone of the repository root:

```
pip install -e ".[viz]"
```

The core depends only on numpy. The `viz` extra adds matplotlib for figures.
Framework-specific code (torch, onnx) is only imported when a check actually
needs it.

## The one thing vibe-check needs

Whatever your model is, reduce it to a `predict` function. It maps a batch of
inputs to a batch of outputs:

```
predict(X) -> y_hat        # numpy in, numpy out, batched
```

Then hand it your data splits. Everything else is optional and only unlocks
more checks.

## Quickstart

This is a complete, runnable script. The "surrogate" here is a linear
least-squares fit, so it needs nothing beyond numpy. The workflow is the same
for a neural network: wrap it in `predict` and pass your splits.

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
print(report)
report.to_markdown("report.md")
report.to_html("report.html")
```

Running it prints (the speed number will vary by machine):

```
vibe-check: PASS
  leakage.normalization: PASS - normalization statistics match the train-only statistics
  leakage.split_overlap: PASS - no duplicate or near-duplicate rows detected across splits
  distribution.coverage: PASS - 1.0% of test points outside the training domain (within tolerance)
  distribution.drift: PASS - train and held-out marginals are close (max KS 0.11)
  error.pointwise: PASS - RMSE 0.09773, skill 0.93
  error.field: SKIP - skipped: output is not a spatial field (field size 2)
  calibration.coverage: SKIP - skipped: no predicted uncertainty (return (mean, std) or set metadata['predicted_std'])
  constraints.physical: SKIP - skipped: need predict, X_test, and metadata['constraints']
  export.roundtrip: SKIP - skipped: need predict, X_test, and metadata['exported_predict']
  speed.inference: PASS - within budget: 96014470 samples/s, 1.04e-08s/sample
```

Every registered check ran. Some passed. The checks that had nothing to work
with returned `SKIP`: a first-class outcome, never a silent pass. Each SKIP
line says what to provide to turn that check on. `report.html` is the same
report as a single self-contained page. Set `metadata["make_figures"] = True`
(with the `viz` extra installed) to embed diagnostic figures in it.

## Reading the report

Each check returns a `CheckResult` with a `status`, a one-line `summary`, and
machine-readable `metrics`. The five statuses:

- `PASS` - the check ran and the surrogate met it.
- `WARN` - worth a look, not necessarily wrong.
- `FAIL` - a real problem the check is confident about.
- `SKIP` - the check could not run (its inputs were not provided). Not a pass.
- `ERROR` - the check itself raised; treated as distinct from FAIL.

`report.summary()` returns the worst status across all checks. Use it to gate
CI:

```python
if report.summary() is vc.Status.FAIL:
    raise SystemExit("surrogate failed validation")
```

## Turning on more checks

Some checks need more than the raw arrays. They read extra settings from the
optional `metadata` dict. A check that does not find what it needs returns
`SKIP`. So you turn on a check simply by providing its inputs:

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

Each check reads its thresholds from `metadata`. You can loosen or tighten a
check without changing code, for example `metadata["drift"] = {"warn_ks": 0.15,
"fail_ks": 0.3}`. To skip a check entirely, just do not provide its inputs. It
returns `SKIP` and stays out of the way.

## Next steps

The [`examples/`](../examples) directory has three worked surrogates
(Robertson kinetics, the Lorenz system, and an FNO on Burgers), each with a
committed report. Use them as templates for your own model.
