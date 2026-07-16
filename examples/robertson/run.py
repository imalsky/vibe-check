"""Robertson stiff chemical kinetics: a local state-to-state emulator.

Generates trajectories of the Robertson three-species system, trains a small
MLP to map a state (plus log step size) to the next state, then runs vibe-check
on the trained surrogate and writes a report next to this file.

The surrogate is intentionally modest: the point is to exercise the reliability
checks (normalization hygiene, coverage, error, physical constraints, export
round-trip, speed), not to build a state-of-the-art integrator.

Run from a clean checkout (after ``pip install -e ".[viz]"`` and the example
requirements):

    python examples/robertson/run.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor

import vibecheck as vc

HERE = pathlib.Path(__file__).resolve().parent
SEED = 0

# Robertson rate constants.
K1, K2, K3 = 0.04, 3.0e7, 1.0e4


def rhs(t, y):
    y1, y2, y3 = y
    return [
        -K1 * y1 + K3 * y2 * y3,
        K1 * y1 - K3 * y2 * y3 - K2 * y2 * y2,
        K2 * y2 * y2,
    ]


def _integrate(y0, t_eval):
    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), y0, method="BDF",
        t_eval=t_eval, rtol=1e-8, atol=1e-12,
    )
    return np.clip(sol.y.T, 0.0, None)  # drop tiny numerical negatives


def _trajectory_pairs(comps, t_eval, log_dt):
    """Concatenated (input, next-state) pairs for a set of initial compositions."""
    xs, ys = [], []
    for y0 in comps:
        states = _integrate(y0, t_eval)
        xs.append(np.column_stack([states[:-1], log_dt]))
        ys.append(states[1:])
    return np.concatenate(xs), np.concatenate(ys)


def make_dataset():
    """State-to-state pairs: input [y1, y2, y3, log10(dt)] -> next [y1, y2, y3].

    Many trajectories are seeded from initial compositions spread across the
    same range, then split by whole trajectory. Splitting by trajectory is the
    correct way to hold out sequential data (two states from one trajectory
    never straddle the boundary), and because every split spans the same range
    of initial states, the train and held-out marginals stay comparable.
    """
    rng = np.random.default_rng(SEED)
    t_eval = np.logspace(-5, 5, 160)
    log_dt = np.log10(np.diff(t_eval))

    n_traj = 40
    y1_0 = rng.uniform(0.5, 1.0, size=n_traj)
    comps = [[a, 0.0, 1.0 - a] for a in y1_0]
    order = rng.permutation(n_traj)
    train = [comps[i] for i in order[:28]]
    val = [comps[i] for i in order[28:34]]
    test = [comps[i] for i in order[34:]]
    return (
        _trajectory_pairs(train, t_eval, log_dt),
        _trajectory_pairs(val, t_eval, log_dt),
        _trajectory_pairs(test, t_eval, log_dt),
    )


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_dataset()

    # Standardize using TRAIN statistics only (correct normalization hygiene).
    x_mu, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    y_mu, y_sd = y_train.mean(axis=0), y_train.std(axis=0)
    x_sd = np.where(x_sd < 1e-12, 1.0, x_sd)
    y_sd = np.where(y_sd < 1e-12, 1.0, y_sd)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs",
        max_iter=2000, random_state=SEED,
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

    metadata = {
        "normalization": {"mean": x_mu, "std": x_sd},
        "constraints": [
            {"type": "positivity", "channels": [0, 1, 2], "name": "species_nonneg"},
            {"type": "sum", "value": 1.0, "axis": -1, "channels": [0, 1, 2],
             "tol": 1e-2, "name": "mass_conservation"},
        ],
        "exported_predict": exported_predict,
        "speed": {"min_throughput": 1.0e4},
        "make_figures": True,
    }

    report = vc.check(
        predict,
        X_train=x_train, y_train=y_train,
        X_val=x_val, y_val=y_val,
        X_test=x_test, y_test=y_test,
        metadata=metadata,
    )
    title = "vibe-check report: Robertson state-to-state MLP"
    report.to_markdown(str(HERE / "report.md"), title=title)
    report.to_html(str(HERE / "report.html"), title=title)
    print(f"Robertson: overall {report.summary().value.upper()} "
          f"over {len(report.results)} checks. Wrote report.md and report.html.")


if __name__ == "__main__":
    main()
