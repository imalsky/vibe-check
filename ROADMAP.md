# Roadmap

Milestones follow the URSSI proposal timeline. Check items off as they land.
Keep the checkboxes honest: a box is only checked when the check is implemented,
tested, and its behavior is documented.

## M1 (Jun-Jul) - Contract and skeleton
- [x] Package skeleton, CI, license, README
- [x] Draft validation contract (`docs/VALIDATION_CONTRACT.md`)
- [ ] Finalize `CheckResult` / `Report` API after first two checks land
- [x] `report.to_html` with embedded figures

## M2 (Jul-Aug) - Split, normalization, distribution, export
- [x] `leakage.normalization`
- [x] `leakage.split_overlap`
- [x] `distribution.coverage`
- [x] `distribution.drift`
- [ ] `export.roundtrip`

## M3 (Aug-Sep) - Error, calibration, constraint, speed
- [x] `error.pointwise`
- [x] `error.field`
- [x] `calibration.coverage`
- [ ] `constraints.physical`
- [ ] `speed.inference`

## M4 (Sep-Nov) - Validate, document, release
- [ ] Robertson example + report template
- [ ] Lorenz example + report template
- [ ] FNO (Burgers) example + report template
- [ ] Tutorial / end-to-end workflow doc
- [ ] Tagged release; archive examples
- [ ] URSSI case study on normalization leakage

## Ongoing (community)
- [ ] Monthly progress note
- [ ] Triage `community-feedback` issues; document changes to the diagnostic set
