# vibe-check report

Overall: **WARN**

## leakage.normalization - PASS
normalization statistics match the train-only statistics

- `rel_distance_to_train_stats`: 0.0
- `rel_distance_to_full_stats`: 0.3957314207401588
- `rtol`: 0.001

## leakage.split_overlap - PASS
no duplicate or near-duplicate rows detected across splits

- `exact_duplicate_rows`: 0.0
- `near_duplicate_rows`: 0.0
- `min_standardized_distance`: 0.1785695514893448
- `atol`: 0.001

- X_train vs X_val: 0 exact, 0 near-duplicate rows
- X_train vs X_test: 0 exact, 0 near-duplicate rows
- X_val vs X_test: 0 exact, 0 near-duplicate rows

## distribution.coverage - PASS
test points are within the training domain

- `frac_test_out_of_domain`: 0.0
- `n_features_with_extrapolation`: 0.0
- `max_feature_out_of_domain_frac`: 0.0
- `max_excursion_in_feature_ranges`: 0.0

## distribution.drift - WARN
moderate marginal drift (max KS 0.11) vs X_test

- `max_ks_distance`: 0.11476190476190476
- `worst_feature_index`: 10.0
- `mean_ks_distance`: 0.06918154761904763
- `warn_ks`: 0.1
- `fail_ks`: 0.2

- train vs X_val: max KS 0.114 at feature 11
- train vs X_test: max KS 0.115 at feature 10

## error.pointwise - PASS
RMSE 0.001574, skill 0.99

- `rmse`: 0.0015744889920434359
- `mae`: 0.001079125597983685
- `max_abs_error`: 0.016037722405824063
- `r2`: 0.9999577225186371
- `skill_vs_mean_baseline`: 0.9935033405368615

## error.field - PASS
mean field error 0.6%, skill 0.99

- `field_rmse`: 0.0015744889920434359
- `mean_abs_percent_error`: 0.6447943062923639
- `max_abs_percent_error`: 4.757832433601773
- `worst_sample_index`: 10.0
- `skill_vs_mean_field`: 0.9935033405368615

## calibration.coverage - WARN
stated uncertainty is loosely calibrated (max gap 0.09); underconfident (intervals too wide)

- `empirical_coverage_1sigma`: 0.775
- `empirical_coverage_2sigma`: 0.9382291666666667
- `empirical_coverage_3sigma`: 0.9780208333333333
- `nominal_coverage_1sigma`: 0.6826894921370859
- `nominal_coverage_2sigma`: 0.9544997361036416
- `max_abs_coverage_deviation`: 0.09231050786291417

## constraints.physical - SKIP
skipped: need predict, X_test, and metadata['constraints']

## export.roundtrip - PASS
exported model matches in-memory model (max abs diff 0)

- `max_abs_difference`: 0.0
- `max_rel_difference`: 0.0
- `fraction_mismatched`: 0.0
- `atol`: 0.0001
- `rtol`: 0.001

## speed.inference - PASS
within budget: 5871 samples/s, 0.00017s/sample

- `median_batch_seconds`: 0.025547875004122034
- `throughput_samples_per_s`: 5871.329806326286
- `latency_per_sample_s`: 0.0001703191666941469
- `n_samples`: 150.0
