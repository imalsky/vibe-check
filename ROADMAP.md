# Roadmap

Milestones follow the URSSI proposal timeline. Check items off as they land.
Keep the checkboxes honest: a box is only checked when the check is implemented,
tested, and its behavior is documented.

## M1 (Jun-Jul) - Contract and skeleton
- [x] Package skeleton, CI, license, README
- [x] Draft validation contract (`docs/VALIDATION_CONTRACT.md`)
- [x] Finalize `CheckResult` / `Report` API after first two checks land
- [x] `report.to_html` with embedded figures

## M2 (Jul-Aug) - Split, normalization, distribution, export
- [x] `leakage.normalization`
- [x] `leakage.split_overlap`
- [x] `distribution.coverage`
- [x] `distribution.drift`
- [x] `export.roundtrip`

## M3 (Aug-Sep) - Error, calibration, constraint, speed
- [x] `error.pointwise`
- [x] `error.field`
- [x] `calibration.coverage`
- [x] `constraints.physical`
- [x] `speed.inference`

## M4 (Sep-Nov) - Validate, document, release
- [x] Robertson example + report template
- [x] Lorenz example + report template
- [x] FNO (Burgers) example + report template
- [x] Tutorial / end-to-end workflow doc
- [ ] Tagged release (0.1.0 prepared: version, CHANGELOG, packaging verified;
  tag and PyPI publish pending sign-off). Examples archived with committed reports.
- [x] URSSI case study on normalization leakage

## Ongoing (community)
- [ ] Monthly progress note
- [ ] Triage `community-feedback` issues; document changes to the diagnostic set
