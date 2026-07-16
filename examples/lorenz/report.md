# vibe-check report

Overall: **WARN**

## leakage.normalization - PASS
normalization statistics match the train-only statistics

- `rel_distance_to_train_stats`: 0.0
- `rel_distance_to_full_stats`: 0.6085786599796111
- `rtol`: 0.001

## leakage.split_overlap - WARN
16 near-duplicate row(s) across splits (standardized distance < 0.001)

- `exact_duplicate_rows`: 0.0
- `near_duplicate_rows`: 16.0
- `min_standardized_distance`: 0.0007607884718584963
- `atol`: 0.001

- X_train vs X_val: 0 exact, 16 near-duplicate rows
- X_train vs X_test: 0 exact, 0 near-duplicate rows
- X_val vs X_test: 0 exact, 0 near-duplicate rows
- note: splits over 3000 rows were subsampled

## distribution.coverage - PASS
test points are within the training domain

- `frac_test_out_of_domain`: 0.0
- `n_features_with_extrapolation`: 0.0
- `max_feature_out_of_domain_frac`: 0.0
- `max_excursion_in_feature_ranges`: 0.0

## distribution.drift - WARN
moderate marginal drift (max KS 0.18) vs X_val

- `max_ks_distance`: 0.1809916537867079
- `worst_feature_index`: 0.0
- `mean_ks_distance`: 0.10253565270259384
- `warn_ks`: 0.1
- `fail_ks`: 0.2

- train vs X_val: max KS 0.181 at feature 0
- train vs X_test: max KS 0.099 at feature 0

## error.pointwise - PASS
RMSE 0.1443, skill 0.98

- `rmse`: 0.14425124154464747
- `mae`: 0.10601489648030922
- `max_abs_error`: 1.0042311741948176
- `r2`: 0.9998877802176956
- `skill_vs_mean_baseline`: 0.9831755726493582

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

## error.field - SKIP
skipped: output is not a spatial field (field size 18)

## calibration.coverage - WARN
stated uncertainty is loosely calibrated (max gap 0.09); underconfident (intervals too wide)

- `empirical_coverage_1sigma`: 0.7739928909952607
- `empirical_coverage_2sigma`: 0.9662980516061085
- `empirical_coverage_3sigma`: 0.9933846761453397
- `nominal_coverage_1sigma`: 0.6826894921370859
- `nominal_coverage_2sigma`: 0.9544997361036416
- `max_abs_coverage_deviation`: 0.09130339885817484

## constraints.physical - SKIP
skipped: need predict, X_test, and metadata['constraints']

## export.roundtrip - PASS
exported model matches in-memory model (max abs diff 3.62e-06)

- `max_abs_difference`: 3.624149272241084e-06
- `max_rel_difference`: 0.002498659722543057
- `fraction_mismatched`: 0.0
- `atol`: 0.0001
- `rtol`: 0.001

## speed.inference - PASS
within budget: 770395 samples/s, 1.3e-06s/sample

- `median_batch_seconds`: 0.0021910829818807542
- `throughput_samples_per_s`: 770395.285782867
- `latency_per_sample_s`: 1.2980349418724847e-06
- `n_samples`: 1688.0
