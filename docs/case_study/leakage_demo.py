"""Runnable demonstration for the normalization-leakage case study.

One feature is nearly constant in the training data but varies at test time (a
common real bug). We fit the same surrogate twice, differing only in how the
input scaler was estimated:

- honest: scaler fit on the training split only (what you can do in deployment);
- leaked: scaler fit on the full dataset, train plus test (leakage).

The leaked pipeline reports a much lower test error, but that number cannot be
reproduced in deployment, where only training statistics are available.
vibe-check's leakage.normalization check flags the leaked scaler automatically.

Run:

    python docs/case_study/leakage_demo.py
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

import vibecheck as vc

SEED = 0


def target(x):
    # Depends on x1 and x2 only; x0 is irrelevant to the target but is the
    # feature whose training variance is tiny.
    return (np.tanh(1.5 * x[:, 1]) - np.tanh(1.5 * x[:, 2])).reshape(-1, 1)


def fit_and_predict(x_train, y_train, mean, std):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
        max_iter=1500, early_stopping=True, random_state=SEED,
    )
    model.fit((x_train - mean) / std, y_train.ravel())

    def predict(x):
        return model.predict((np.asarray(x) - mean) / std).reshape(-1, 1)

    return predict


def main():
    rng = np.random.default_rng(SEED)
    # x0: nearly constant in train (std 0.03), but varies at test time.
    x_train = np.column_stack([
        rng.normal(0.0, 0.03, 2000),
        rng.normal(0.0, 1.0, 2000),
        rng.normal(0.0, 1.0, 2000),
    ])
    x_test = np.column_stack([
        rng.normal(0.0, 1.0, 1000),
        rng.normal(0.0, 1.0, 1000),
        rng.normal(0.0, 1.0, 1000),
    ])
    y_train, y_test = target(x_train), target(x_test)

    train_mean, train_std = x_train.mean(0), x_train.std(0)
    full = np.concatenate([x_train, x_test], axis=0)
    full_mean, full_std = full.mean(0), full.std(0)

    def rmse(predict):
        return float(np.sqrt(np.mean((predict(x_test) - y_test) ** 2)))

    honest = fit_and_predict(x_train, y_train, train_mean, train_std)
    leaked = fit_and_predict(x_train, y_train, full_mean, full_std)
    honest_rmse, leaked_rmse = rmse(honest), rmse(leaked)

    print(f"train per-feature std: {np.round(train_std, 3)}")
    print(f"full  per-feature std: {np.round(full_std, 3)}")
    print(f"honest (train-only scaler) test RMSE: {honest_rmse:.4f}")
    print(f"leaked (full-data scaler)  test RMSE: {leaked_rmse:.4f}")
    print(f"leaked appears {honest_rmse / leaked_rmse:.1f}x more accurate")

    for label, predict, (mean, std) in [
        ("honest", honest, (train_mean, train_std)),
        ("leaked", leaked, (full_mean, full_std)),
    ]:
        report = vc.check(
            predict, X_train=x_train, X_test=x_test, y_test=y_test,
            metadata={"normalization": {"mean": mean, "std": std}},
        )
        result = next(r for r in report.results if r.name == "leakage.normalization")
        print(f"{label}: leakage.normalization -> {result.status.value.upper()}")


if __name__ == "__main__":
    main()
