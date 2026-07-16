# Examples

Three small surrogates that cover the common shapes of scientific ML, used to
validate `vibe-check` and to serve as report templates other groups can adapt.

Each example directory will contain: a script that trains a small surrogate, a
script that runs `vibe-check` on it, and the resulting report checked in as a
template. Data and trained weights are gitignored; the scripts regenerate them.

## robertson/ - local state-to-state emulator

The Robertson stiff chemical-kinetics problem: a canonical three-species ODE
with widely separated reaction timescales. The surrogate maps a state to the
next state. Good stress test for positivity and mass-conservation constraints.

## lorenz/ - ordered-output trajectory emulator

The Lorenz system: the surrogate predicts a time-ordered state trajectory on a
fixed output grid. Exercises the error and coverage checks on sequential output.

## fno_burgers/ - spatial-field emulator

A Fourier Neural Operator on a standard benchmark (Burgers, extendable to Darcy
flow and Navier-Stokes). Exercises the field-level error maps and the
distribution-coverage checks on function-space data.

Deliberately built to run and document within the fellowship, but broad enough
to show the validation setup on local, sequential, and spatial-field models.
