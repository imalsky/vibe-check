"""Lorenz system: an ordered-output trajectory emulator.

Trains a small MLP to map a Lorenz state to the next few states on a fixed time
grid (a short ordered trajectory), then runs vibe-check on it. A per-channel
predictive standard deviation is estimated from validation residuals, so this
example also exercises the calibration check. Writes a report next to this file.

Run:

    python examples/lorenz/run.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.style
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor

import vibecheck as vc

HERE = pathlib.Path(__file__).resolve().parent

# Report figures follow the shared example style; remove these two lines to
# use your own matplotlib defaults.
matplotlib.use("Agg")
matplotlib.style.use(HERE.parent / "science.mplstyle")
SEED = 0

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.02
HORIZON = 6  # number of future states the surrogate predicts


def rhs(t, s):
    x, y, z = s
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def make_windows():
    """input = state s(t); output = the next HORIZON states, flattened.

    One long trajectory is cut into contiguous train / val / test blocks with a
    gap of HORIZON between them, so no prediction window straddles a split
    boundary (the correct hold-out for a single sequence).
    """
    n_steps = 8500
    t_eval = np.arange(n_steps) * DT
    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), [1.0, 1.0, 1.0], method="RK45",
        t_eval=t_eval, rtol=1e-9, atol=1e-9,
    )
    states = sol.y.T[500:]  # drop transient onto the attractor

    n = states.shape[0] - HORIZON
    x = states[:n]
    y = np.stack([states[i + 1 : i + 1 + HORIZON] for i in range(n)])
    y = y.reshape(n, HORIZON * 3)

    gap = HORIZON
    tr_end = 5000
    va_start, va_end = tr_end + gap, 6300
    te_start = va_end + gap
    return (
        (x[:tr_end], y[:tr_end]),
        (x[va_start:va_end], y[va_start:va_end]),
        (x[te_start:n], y[te_start:n]),
    )


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_windows()

    x_mu, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    y_mu, y_sd = y_train.mean(axis=0), y_train.std(axis=0)
    x_sd = np.where(x_sd < 1e-12, 1.0, x_sd)
    y_sd = np.where(y_sd < 1e-12, 1.0, y_sd)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 128), activation="tanh", solver="adam",
        max_iter=800, early_stopping=True, n_iter_no_change=20,
        random_state=SEED,
    )
    model.fit((x_train - x_mu) / x_sd, (y_train - y_mu) / y_sd)

    def predict(x_raw):
        scaled = (np.asarray(x_raw, dtype=float) - x_mu) / x_sd
        return model.predict(scaled) * y_sd + y_mu

    def exported_predict(x_raw):
        # Simulate an export round-trip by rounding the standardized inputs
        # through float32 (as an ONNX pipeline would); the model itself still
        # runs in float64, so the diff is small but non-zero.
        scaled = ((np.asarray(x_raw, dtype=np.float32) - x_mu.astype(np.float32))
                  / x_sd.astype(np.float32))
        return model.predict(scaled.astype(float)) * y_sd + y_mu

    # Per-output predictive std from validation residuals (a simple, honest
    # uncertainty estimate), broadcast across the test set for calibration.
    resid_val = predict(x_val) - y_val
    sigma_channel = resid_val.std(axis=0)
    predicted_std = np.broadcast_to(sigma_channel, y_test.shape).copy()

    metadata = {
        "normalization": {"mean": x_mu, "std": x_sd},
        "predicted_std": predicted_std,
        "exported_predict": exported_predict,
        "speed": {"min_throughput": 1.0e4},
        # Ordered trajectory output, not a spatial field.
        "error_field": {"is_field": False},
        "make_figures": True,
    }

    report = vc.check(
        predict,
        X_train=x_train, y_train=y_train,
        X_val=x_val, y_val=y_val,
        X_test=x_test, y_test=y_test,
        metadata=metadata,
    )
    title = "vibe-check report: Lorenz trajectory MLP"
    report.to_markdown(str(HERE / "report.md"), title=title)
    report.to_html(str(HERE / "report.html"), title=title)
    print(f"Lorenz: overall {report.summary().value.upper()} "
          f"over {len(report.results)} checks. Wrote report.md and report.html.")


if __name__ == "__main__":
    main()
