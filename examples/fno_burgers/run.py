"""Fourier Neural Operator on 1-D Burgers: a spatial-field emulator.

Generates a Burgers dataset (initial condition -> solution at a fixed time) with
a pseudo-spectral solver, trains a small FNO to learn that operator, and runs
vibe-check on it. Because the output is a spatial field this example exercises
the field-level error maps (error.field) in addition to the other checks.

Run (needs torch; see requirements.txt):

    python examples/fno_burgers/run.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib
import matplotlib.style
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import vibecheck as vc

HERE = pathlib.Path(__file__).resolve().parent

# Report figures follow the shared example style; remove these two lines to
# use your own matplotlib defaults.
matplotlib.use("Agg")
matplotlib.style.use(HERE.parent / "science.mplstyle")
SEED = 0
GRID = 64


def make_burgers(n_samples, s=GRID, nu=0.02, t_final=0.5, dt=1e-3, seed=SEED):
    """Solve viscous Burgers on a periodic domain from smooth random ICs.

    Returns ``(u0, uT)``: the initial condition and the solution at ``t_final``,
    each sampled on ``s`` grid points. Pseudo-spectral in space, RK4 in time,
    vectorized over all samples at once.
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(s, d=1.0 / s)
    k = 2.0 * np.pi * freqs

    amplitude = (np.abs(freqs) + 1.0) ** (-2.0)
    coeffs = (rng.normal(size=(n_samples, s)) + 1j * rng.normal(size=(n_samples, s)))
    coeffs = coeffs * amplitude
    coeffs[:, 0] = 0.0
    u0 = np.fft.ifft(coeffs, axis=1).real
    u0 = u0 / (u0.std(axis=1, keepdims=True) + 1e-8) * 0.5

    def rhs(u):
        u_hat = np.fft.fft(u, axis=1)
        u_x = np.fft.ifft(1j * k * u_hat, axis=1).real
        u_xx = np.fft.ifft(-(k**2) * u_hat, axis=1).real
        return -u * u_x + nu * u_xx

    u = u0.copy()
    for _ in range(int(round(t_final / dt))):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u0, u


class SpectralConv1d(nn.Module):
    """One FNO spectral convolution: keep the lowest ``modes`` Fourier modes."""

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.out_ch = out_ch
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        b, _, s = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            b, self.out_ch, s // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=s)


class FNO1d(nn.Module):
    """A small 1-D Fourier Neural Operator."""

    def __init__(self, modes=16, width=32, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(2, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 2, 1)
        for spectral, pointwise in zip(self.spectral, self.pointwise, strict=True):
            x = F.gelu(spectral(x) + pointwise(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def _features(u_std, xgrid):
    b, s = u_std.shape
    xg = xgrid.unsqueeze(0).expand(b, s)
    return torch.stack([u_std, xg], dim=-1)


def main():
    torch.manual_seed(SEED)
    u0, uT = make_burgers(1000)
    x_train, y_train = u0[:700], uT[:700]
    x_val, y_val = u0[700:850], uT[700:850]
    x_test, y_test = u0[850:], uT[850:]

    x_mu, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    y_mu, y_sd = y_train.mean(axis=0), y_train.std(axis=0)
    x_sd = np.where(x_sd < 1e-12, 1.0, x_sd)
    y_sd = np.where(y_sd < 1e-12, 1.0, y_sd)

    xgrid = torch.tensor(np.linspace(0.0, 1.0, GRID, endpoint=False), dtype=torch.float32)
    model = FNO1d(modes=16, width=32, layers=4)

    x_tr = torch.tensor((x_train - x_mu) / x_sd, dtype=torch.float32)
    y_tr = torch.tensor((y_train - y_mu) / y_sd, dtype=torch.float32)
    feats_tr = _features(x_tr, xgrid)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch = 25
    for _ in range(80):
        perm = torch.randperm(x_tr.shape[0])
        for start in range(0, x_tr.shape[0], batch):
            idx = perm[start : start + batch]
            opt.zero_grad()
            loss = loss_fn(model(feats_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
    model.eval()

    def predict(x_raw):
        u_std = (np.asarray(x_raw, dtype=float) - x_mu) / x_sd
        with torch.no_grad():
            out = model(_features(torch.tensor(u_std, dtype=torch.float32), xgrid))
        return out.numpy() * y_sd + y_mu

    # A realistic export round-trip: TorchScript trace vs eager, both float32.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # benign fixed-shape tracer warnings
        traced = torch.jit.trace(model, _features(x_tr[:4], xgrid))

    def exported_predict(x_raw):
        u_std = (np.asarray(x_raw, dtype=float) - x_mu) / x_sd
        with torch.no_grad():
            out = traced(_features(torch.tensor(u_std, dtype=torch.float32), xgrid))
        return out.numpy() * y_sd + y_mu

    resid_val = predict(x_val) - y_val
    predicted_std = np.broadcast_to(resid_val.std(axis=0), y_test.shape).copy()

    metadata = {
        "normalization": {"mean": x_mu, "std": x_sd},
        # Field inputs extrapolate slightly per grid point; allow a 10% margin
        # on the training envelope before counting a point out of domain.
        "coverage": {"margin": 0.1},
        "predicted_std": predicted_std,
        "exported_predict": exported_predict,
        "speed": {"min_throughput": 1.0e2},
        "error_field": {"is_field": True},
        "make_figures": True,
    }

    report = vc.check(
        predict,
        X_train=x_train, y_train=y_train,
        X_val=x_val, y_val=y_val,
        X_test=x_test, y_test=y_test,
        metadata=metadata,
    )
    title = "vibe-check report: FNO on 1-D Burgers"
    report.to_markdown(str(HERE / "report.md"), title=title)
    report.to_html(str(HERE / "report.html"), title=title)
    print(f"FNO/Burgers: overall {report.summary().value.upper()} "
          f"over {len(report.results)} checks. Wrote report.md and report.html.")


if __name__ == "__main__":
    main()
