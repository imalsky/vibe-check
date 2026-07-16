# Examples

Three small surrogates that cover the common shapes of scientific ML, used to
validate `vibe-check` and to serve as report templates other groups can adapt.
Each directory has a self-contained `run.py` that generates its data, trains a
small surrogate, runs `vibe-check`, and writes `report.md` and `report.html`
next to the script. The generated reports are checked in as templates; the raw
data and trained weights are not (the scripts regenerate them from a fixed
seed). Results are reproducible on a given machine, but bitwise identity across
platforms is not guaranteed, and the timing numbers in the reports vary by
machine.

## Running

From the repository root, install the package with the plotting extra and the
example's requirements, then run its script:

```
pip install -e ".[viz]"
pip install -r examples/robertson/requirements.txt
python examples/robertson/run.py
```

The same pattern applies to `examples/lorenz` and `examples/fno_burgers` (the
FNO example needs PyTorch).

Report figures use the shared matplotlib style in `examples/science.mplstyle`;
each script applies it near the top, and removing those lines falls back to
your own matplotlib defaults. The checks draw with cycle-relative colors, so
they follow whatever style is active.

## robertson/ - local state-to-state emulator

The Robertson stiff chemical-kinetics problem: a three-species ODE with widely
separated timescales. An MLP maps a state (plus log step size) to the next
state. Many trajectories are split by whole trajectory, the correct hold-out for
sequential data. Highlights the normalization, mass-conservation, and positivity
checks. Committed report: overall WARN (skill 0.99, with cautions on
near-duplicate states across the trajectory splits and mild drift).

## lorenz/ - ordered-output trajectory emulator

The Lorenz system: an MLP predicts the next several states on a fixed time grid.
A per-channel predictive standard deviation is estimated from validation
residuals, so this example also exercises `calibration.coverage`. Committed
report: overall WARN (skill 0.98; cautions on the constant per-channel
uncertainty being underconfident, a handful of near-duplicate rows across the
gapped splits, and mild drift between the trajectory segments).

## fno_burgers/ - spatial-field emulator

A Fourier Neural Operator learning the 1-D viscous Burgers solution operator
(initial condition -> solution at a fixed time). Because the output is a spatial
field, this example exercises the field-level error maps (`error.field`) and a
real export round-trip (TorchScript trace vs eager). Committed report: overall
WARN (0.6% mean field error, skill 0.99, with a calibration caution).

All three are deliberately small enough to run and document within the
fellowship, but broad enough to show the validation setup on local, sequential,
and spatial-field models.
