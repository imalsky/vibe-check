# vibe-check: fellowship timeline

| | |
|---|---|
| **Project** | Reliability Checks for Machine-Learning Surrogates in Scientific Software |
| **Fellow** | Isaac Malsky |
| **Period** | July 2026 through December 2026 |
| **Repository** | https://github.com/imalsky/vibe-check |
| **Updated** | July 28, 2026 |

This is the timeline deliverable for the 2026 URSSI Early-Career Fellowship. It
maps the proposal (`docs/proposal/URSSI.pdf`) onto the fellowship period and
records current status. `ROADMAP.md` tracks individual work items and updates as
work lands. This document is the higher-level view.

## Status as of July 28, 2026

The diagnostic core is complete. All ten checks in the validation contract are
implemented, registered, tested, and documented. The three worked examples run
end to end, with committed reports. The proposal spread this work across four
months, ending in November. It was actually finished in July.

An early package is not a finished project. The proposal treated the remaining
work as secondary. I now think it is the harder half: putting the package in
front of people who build emulators in other fields, and changing the checks
based on what they find. A validation tool built on one person's assumptions is
not a shared standard.

| Item | Status |
|---|---|
| Validation contract (`docs/VALIDATION_CONTRACT.md`) | Done |
| Package skeleton, CI (Python 3.10 / 3.11 / 3.12), MIT | Done |
| Ten checks implemented, registered, tested | Done |
| `report.to_markdown` and `report.to_html` | Done |
| Robertson, Lorenz, FNO (Burgers) examples with reports | Done |
| Tutorial and end-to-end workflow doc | Done |
| Case study on normalization leakage | Done |
| Kickoff meeting | Done (July 23) |
| Public repository | Pending |
| Tagged 0.1.0 release and PyPI publish | Pending |

## Plan, July to December 2026

### July (complete)

Define the validation contract and publish the package skeleton. Implement all
ten checks: `leakage.normalization`, `leakage.split_overlap`,
`distribution.coverage`, `distribution.drift`, `error.pointwise`, `error.field`,
`calibration.coverage`, `constraints.physical`, `export.roundtrip`, and
`speed.inference`. Build the three representative examples. Write the tutorial
and the normalization-leakage case study. Attend the fellowship kickoff meeting.

### August

Make the repository public and tag the 0.1.0 release, then publish to PyPI. Post
the introduction blog entry to the URSSI site and send this timeline (both due
August 1). Begin bi-weekly URSSI check-ins on August 14. Open the
`community-feedback` issue label and begin recruiting reviewers, starting with
the AI/ML group at JPL and with researchers building emulators in climate,
computational chemistry, and materials.

### September and October

Run the package against surrogates I did not write. This is the real test: can
someone who did not train the model read the report? I expect it to change the
default thresholds and the wording of the summaries. Triage community feedback
and document every change it causes to the checks. A check that changes because
of outside feedback is a better outcome for this fellowship than a check I
happened to guess right. Post monthly progress notes in the repository. Publish
the update blog entry, covering progress and the most interesting failure found
so far.

### November

Validate the package in a second scientific domain. The proposal promises
guidance that works across at least two domains, so this is where that claim
gets tested. Archive the examples with their reports, so every number in them
can be reproduced from a clean checkout. Make any API changes that outside
feedback requires, before a stable release.

### December

Publish the conclusion blog entry. Write the final report (about 1,500 words),
covering what was proposed, what was delivered, how the work affected the
field, and what comes next. Cut a final tagged release, with documentation and
examples up to date.

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
above depends on it, so making the repository public is the first task in
August.

**Community feedback may not arrive.** Asking for outside input is easy to plan
and hard to actually get. If the `community-feedback` label is still empty by
late September, I will switch strategy: directly recruit two or three emulator
authors and watch them run the package, instead of waiting for people to file
issues on their own.

**Thresholds reflect one person's calibration.** The current default settings
encode my own judgment about what counts as a warning. This is the weakest part
of the package, and the part most likely to change.

## Acknowledgment

This work was supported by the US Research Software Sustainability Institute
(URSSI) via grant G-2022-19347 from the Sloan Foundation.
