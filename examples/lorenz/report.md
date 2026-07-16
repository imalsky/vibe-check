# vibe-check report: Lorenz trajectory MLP

Overall: **WARN**

| check | status | summary |
| --- | --- | --- |
| leakage.split_overlap | WARN | 16 near-duplicate row(s) across splits (standardized distance < 0.001) |
| distribution.drift | WARN | moderate marginal drift (max KS 0.18) vs X_val |
| calibration.coverage | WARN | stated uncertainty is loosely calibrated (max gap 0.09); underconfident (intervals too wide) |
| leakage.normalization | PASS | normalization statistics match the train-only statistics |
| distribution.coverage | PASS | 0.0% of test points outside the training domain (within tolerance) |
| error.pointwise | PASS | RMSE 0.1443, skill 0.98 |
| export.roundtrip | PASS | exported model matches in-memory model (max abs diff 3.62e-06) |
| speed.inference | PASS | within budget: 807800 samples/s, 1.24e-06s/sample |
| error.field | SKIP | skipped: output is not a spatial field (field size 18) |
| constraints.physical | SKIP | skipped: need predict, X_test, and metadata['constraints'] |

## leakage.normalization - PASS
normalization statistics match the train-only statistics

- `rel_distance_to_train_stats`: 0
- `rel_distance_to_full_stats`: 0.01109
- `rtol`: 0.001

## leakage.split_overlap - WARN
16 near-duplicate row(s) across splits (standardized distance < 0.001)

- `exact_duplicate_rows`: 0
- `near_duplicate_rows`: 16
- `min_standardized_distance`: 0.0007608
- `atol`: 0.001

- X_train vs X_val: 0 exact, 16 near-duplicate rows
- X_train vs X_test: 0 exact, 0 near-duplicate rows
- X_val vs X_test: 0 exact, 0 near-duplicate rows
- note: splits over 3000 rows were subsampled

## distribution.coverage - PASS
0.0% of test points outside the training domain (within tolerance)

- `frac_test_out_of_domain`: 0
- `n_features_with_extrapolation`: 0
- `max_feature_out_of_domain_frac`: 0
- `max_excursion_in_feature_ranges`: 0
- `warn_frac`: 0.01
- `fail_frac`: 0.1

(1 figure attached; see the HTML report)

## distribution.drift - WARN
moderate marginal drift (max KS 0.18) vs X_val

- `max_ks_distance`: 0.181
- `worst_feature_index`: 0
- `mean_ks_distance`: 0.1025
- `warn_ks`: 0.1
- `fail_ks`: 0.2

- train vs X_val: max KS 0.181 at feature 0
- train vs X_test: max KS 0.099 at feature 0

(2 figures attached; see the HTML report)

## error.pointwise - PASS
RMSE 0.1443, skill 0.98

- `rmse`: 0.1443
- `mae`: 0.106
- `max_abs_error`: 1.004
- `r2`: 0.9999
- `skill_vs_mean_baseline`: 0.9832
- `warn_skill`: 0.5

Per-channel RMSE:
- channel 0: 0.07748
- channel 1: 0.1434
- channel 2: 0.1494
- channel 3: 0.1198
- channel 4: 0.1395
- channel 5: 0.1203
- channel 6: 0.09636
- channel 7: 0.1664
- channel 8: 0.1438
- channel 9: 0.1072
- channel 10: 0.1551
- channel 11: 0.108
- channel 12: 0.1026
- channel 13: 0.1683
- channel 14: 0.1322
- channel 15: 0.1209
- channel 16: 0.2771
- channel 17: 0.1576

(2 figures attached; see the HTML report)

## error.field - SKIP
skipped: output is not a spatial field (field size 18)

## calibration.coverage - WARN
stated uncertainty is loosely calibrated (max gap 0.09); underconfident (intervals too wide)

- `empirical_coverage_1sigma`: 0.774
- `empirical_coverage_2sigma`: 0.9663
- `empirical_coverage_3sigma`: 0.9934
- `nominal_coverage_1sigma`: 0.6827
- `nominal_coverage_2sigma`: 0.9545
- `nominal_coverage_3sigma`: 0.9973
- `max_abs_coverage_deviation`: 0.0913
- `warn_tol`: 0.07
- `fail_tol`: 0.15

(1 figure attached; see the HTML report)

## constraints.physical - SKIP
skipped: need predict, X_test, and metadata['constraints']

## export.roundtrip - PASS
exported model matches in-memory model (max abs diff 3.62e-06)

- `max_abs_difference`: 3.624e-06
- `max_rel_difference`: 0.002499
- `fraction_mismatched`: 0
- `atol`: 0.0001
- `rtol`: 0.001

## speed.inference - PASS
within budget: 807800 samples/s, 1.24e-06s/sample

- `median_batch_seconds`: 0.00209
- `throughput_samples_per_s`: 8.078e+05
- `latency_per_sample_s`: 1.238e-06
- `n_samples`: 1688
- `min_throughput`: 10000
