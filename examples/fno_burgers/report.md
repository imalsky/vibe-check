# vibe-check report: FNO on 1-D Burgers

Overall: **WARN**

| check | status | summary |
| --- | --- | --- |
| calibration.coverage | WARN | stated uncertainty is loosely calibrated (max gap 0.09); underconfident (intervals too wide) |
| leakage.normalization | PASS | normalization statistics match the train-only statistics |
| leakage.split_overlap | PASS | no duplicate or near-duplicate rows detected across splits |
| distribution.coverage | PASS | 0.0% of test points outside the training domain (within tolerance) |
| distribution.drift | PASS | train and held-out marginals are close (max KS 0.11) |
| error.pointwise | PASS | RMSE 0.001574, skill 0.99 |
| error.field | PASS | mean field error 0.6%, skill 0.99 |
| export.roundtrip | PASS | exported model matches in-memory model (max abs diff 0) |
| speed.inference | PASS | within budget: 6992 samples/s, 0.000143s/sample |
| constraints.physical | SKIP | skipped: need predict, X_test, and metadata['constraints'] |

## leakage.normalization - PASS
normalization statistics match the train-only statistics

- `rel_distance_to_train_stats`: 0
- `rel_distance_to_full_stats`: 0.00785
- `rtol`: 0.001

## leakage.split_overlap - PASS
no duplicate or near-duplicate rows detected across splits

- `exact_duplicate_rows`: 0
- `near_duplicate_rows`: 0
- `min_standardized_distance`: 0.1786
- `atol`: 0.001

- X_train vs X_val: 0 exact, 0 near-duplicate rows
- X_train vs X_test: 0 exact, 0 near-duplicate rows
- X_val vs X_test: 0 exact, 0 near-duplicate rows

## distribution.coverage - PASS
0.0% of test points outside the training domain (within tolerance)

- `frac_test_out_of_domain`: 0
- `n_features_with_extrapolation`: 0
- `max_feature_out_of_domain_frac`: 0
- `max_excursion_in_feature_ranges`: 0
- `warn_frac`: 0.2772
- `fail_frac`: 0.5545

(1 figure attached; see the HTML report)

## distribution.drift - PASS
train and held-out marginals are close (max KS 0.11)

- `max_ks_distance`: 0.1148
- `worst_feature_index`: 10
- `mean_ks_distance`: 0.06918
- `warn_ks`: 0.153
- `fail_ks`: 0.3059

- train vs X_val: max KS 0.114 at feature 11
- train vs X_test: max KS 0.115 at feature 10

(2 figures attached; see the HTML report)

## error.pointwise - PASS
RMSE 0.001574, skill 0.99

- `rmse`: 0.001574
- `mae`: 0.001079
- `max_abs_error`: 0.01604
- `r2`: 1
- `skill_vs_mean_baseline`: 0.9935
- `warn_skill`: 0.5

(2 figures attached; see the HTML report)

## error.field - PASS
mean field error 0.6%, skill 0.99

- `field_rmse`: 0.001574
- `mean_abs_percent_error`: 0.6448
- `max_abs_percent_error`: 4.758
- `worst_sample_index`: 107
- `skill_vs_mean_field`: 0.9935
- `warn_pct`: 5
- `fail_pct`: 20

(1 figure attached; see the HTML report)

## calibration.coverage - WARN
stated uncertainty is loosely calibrated (max gap 0.09); underconfident (intervals too wide)

- `empirical_coverage_1sigma`: 0.775
- `empirical_coverage_2sigma`: 0.9382
- `empirical_coverage_3sigma`: 0.978
- `nominal_coverage_1sigma`: 0.6827
- `nominal_coverage_2sigma`: 0.9545
- `nominal_coverage_3sigma`: 0.9973
- `max_abs_coverage_deviation`: 0.09231
- `warn_tol`: 0.07
- `fail_tol`: 0.15

(1 figure attached; see the HTML report)

## constraints.physical - SKIP
skipped: need predict, X_test, and metadata['constraints']

## export.roundtrip - PASS
exported model matches in-memory model (max abs diff 0)

- `max_abs_difference`: 0
- `max_rel_difference`: 0
- `fraction_mismatched`: 0
- `atol`: 0.0001
- `rtol`: 0.001

## speed.inference - PASS
within budget: 6992 samples/s, 0.000143s/sample

- `median_batch_seconds`: 0.02145
- `throughput_samples_per_s`: 6992
- `latency_per_sample_s`: 0.000143
- `n_samples`: 150
- `min_throughput`: 100
