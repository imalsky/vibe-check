# vibe-check: fellowship timeline

| | |
|---|---|
| **Project** | Vibe-Check: Reliability Checks for Machine-Learning Surrogates in Scientific Software |
| **Fellow** | Isaac Malsky |
| **Period** | July 2026 through December 2026 |
| **Repository** | https://github.com/imalsky/vibe-check |
| **Updated** | July 31, 2026 |

This is the timeline deliverable for the 2026 URSSI Early-Career Fellowship.

## Status as of July 31, 2026

The diagnostic core is in progress. All ten checks in the validation contract
are implemented and passing CI, and the three worked examples run end to end
with committed reports.

An early package is not a finished project. The proposal treated the remaining
work as secondary. I now think it is the harder half: putting the package in
front of people who build emulators in other fields, and changing the checks
based on what they find. A validation tool built on one person's assumptions is
not a shared standard.

| Item | Status |
|---|---|
| Validation contract (`docs/VALIDATION_CONTRACT.md`) | In progress |
| Package skeleton, CI (Python 3.10 / 3.11 / 3.12), MIT | In progress |
| Ten checks implemented, registered, tested | In progress |
| `report.to_markdown` and `report.to_html` | In progress |
| Robertson, Lorenz, FNO (Burgers) examples with reports | In progress |
| Tutorial and end-to-end workflow doc | In progress |
| Case study on normalization leakage | In progress |
| Kickoff meeting | Done (July 23) |
| Public repository | Pending |
| Tagged 0.1.0 release and PyPI publish | Pending |

## Plan, July to December 2026

### July

Define the validation contract and build the package skeleton. Implement all
ten checks: `leakage.normalization`, `leakage.split_overlap`,
`distribution.coverage`, `distribution.drift`, `error.pointwise`, `error.field`,
`calibration.coverage`, `constraints.physical`, `export.roundtrip`, and
`speed.inference`. Build the three representative examples, the tutorial, and
the normalization-leakage case study. Attend the fellowship kickoff meeting.

### August

Make the repository public and tag the 0.1.0 release, then publish to PyPI.
Post the introduction blog entry and send this timeline, both due August 1.
Bi-weekly URSSI check-ins start August 14. Open the `community-feedback` issue
label and start recruiting reviewers: the AI/ML group at JPL first, then
researchers building emulators in climate, computational chemistry, and
materials.

### September and October

Run the package against surrogates I did not write. That is the real test: can
someone who did not train the model actually read the report? I expect it to
change the default thresholds and the wording of the summaries. Triage
community feedback and write down every change it causes. A check that changes
because of outside feedback is a better outcome here than a check I happened to
guess right the first time. Post monthly progress notes in the repository, and
publish the update blog entry: progress plus the most interesting failure found
so far.

### November

Validate the package in a second scientific domain. The proposal promises
guidance that works across at least two domains, so this is where that claim
actually gets tested. Archive the examples with their reports, so every number
in them can be reproduced from a clean checkout. Make any API changes outside
feedback forces, before locking in a stable release.

### December

Publish the conclusion blog entry. Write the final report, about 1,500 words:
what I proposed, what I delivered, how it affected the field, and what comes
next. Cut a final tagged release, with documentation and examples current.

## Deliverables

From the proposal:

1. An open-source Python package that tests normalization, data-split hygiene,
   training-domain coverage, constraint violations, calibration, exported
   inference, and speed.
2. Validation on representative local, sequential, and spatial-field emulators.
3. The package, documentation, and tutorial material open-sourced for community
   use.

From the URSSI fellowship:

4. This timeline (August 1).
5. Three blog posts: introduction (August 1), update, conclusion.
6. Final report, approximately 1,500 words, linking the openly accessible
   products.
7. Bi-weekly check-in meetings from August 14.

## Risks

**The repository is still private.** The proposal promises to host the package
on GitHub from day one. That has not happened yet. Every community-facing item
above depends on it, so making the repository public is the first thing on my
list for August.

**Community feedback may not arrive.** Asking for outside input is easy to plan
and hard to actually get. If the `community-feedback` label is still empty by
late September, I will switch strategy: go recruit two or three emulator
authors directly and watch them run the package, instead of waiting for people
to file issues on their own.

**Thresholds reflect one person's calibration.** The current defaults encode my
own judgment about what counts as a warning. That is the weakest part of the
package, and the part most likely to change.

## Acknowledgment

This work was supported by the US Research Software Sustainability Institute
(URSSI) via grant G-2022-19347 from the Sloan Foundation.
