# vibe-check: fellowship timeline

| | |
|---|---|
| **Project** | Vibe-Check: Reliability Checks for Machine-Learning Surrogates in Scientific Software |
| **Fellow** | Isaac Malsky |
| **Period** | July 2026 through December 2026 |
| **Repository** | https://github.com/imalsky/vibe-check |
| **Updated** | July 31, 2026 |

The timeline of deliverables for the 2026 URSSI Early-Career Fellowship.

## Plan, July to December 2026

This is the overall plan for the codebase over the course of the fellowship.
This work will also be done in coordination with a series of blog posts
documenting the fellowship work for URSSI.

### July (Current)

Define the scope of the validation and surrogate model checks, establish the
package structure and report format, and create a small set of
machine-learning emulators, starting with the three benchmarks the proposal
names. The three toy models will be the Robertson stiff-kinetics system
(Robertson 1966), the Lorenz trajectory (Lorenz 1963), and a Fourier neural
operator on Burgers' equation (Li et al. 2021; Kovachki et al. 2023). These
models are used to test common failure modes of machine-learning
surrogates. Also plan out and start implementing the automated checks. This
is in progress and will be updated continuously in the GitHub repository
above. Submit the introductory blog post and initial timeline by August 1.

### August

Implement the split, normalization, distribution, and export checks. The
current plan covers five checks: `leakage.normalization`
catches scaler statistics fit on test or full data instead of train alone;
`leakage.split_overlap` catches duplicate or near-duplicate rows across train
and test; `distribution.coverage` flags test inputs that fall outside the
training domain; `distribution.drift` flags marginal drift between train,
validation, and test; and `export.roundtrip` checks that the exported model's
output matches the in-memory model.

### September

Add error, calibration, constraint, and speed diagnostics. Improve the
clarity, accessibility, and consistency of generated validation reports.
Complete and validate the three representative benchmark surrogates, and add
further examples only if time permits.

### October

Tag the first public release (0.1.0) and publish it to PyPI so the package
is pip-installable. Put the tutorial, validation contract, and example
reports up on GitHub as the project's documentation, and start soliciting
community suggestions. Submit a progress/update blog post (date
provisional).

### November

Complete and polish the tutorial and three example validation reports.
Continue polishing the project, and iterate on diagnostics and the rest of
the codebase.

### December

Wrap up remaining work, and submit the conclusion blog post and final
report (dates provisional).

## Deliverables

1. An open-source Python package that tests normalization, data-split hygiene,
   training-domain coverage, constraint violations, calibration, exported
   inference, and speed.
2. Validation on representative local, sequential, and spatial-field emulators.
3. The package, documentation, and tutorial material open-sourced for community
   use.

## Acknowledgment

This work was supported by the US Research Software Sustainability Institute
(URSSI) via grant G-2022-19347 from the Sloan Foundation.
