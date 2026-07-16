# vibe-check report

Overall: **WARN**

## leakage.normalization - PASS
normalization statistics match the train-only statistics

- `rel_distance_to_train_stats`: 0.0
- `rel_distance_to_full_stats`: 0.021815152985969307
- `rtol`: 0.001

## leakage.split_overlap - WARN
1401 near-duplicate row(s) across splits (standardized distance < 0.001)

- `exact_duplicate_rows`: 0.0
- `near_duplicate_rows`: 1401.0
- `min_standardized_distance`: 9.306721736502698e-08
- `atol`: 0.001

- X_train vs X_val: 0 exact, 603 near-duplicate rows
- X_train vs X_test: 0 exact, 617 near-duplicate rows
- X_val vs X_test: 0 exact, 181 near-duplicate rows
- note: splits over 3000 rows were subsampled

## distribution.coverage - PASS
test points are within the training domain

- `frac_test_out_of_domain`: 0.0
- `n_features_with_extrapolation`: 0.0
- `max_feature_out_of_domain_frac`: 0.0
- `max_excursion_in_feature_ranges`: 0.0

## distribution.drift - WARN
moderate marginal drift (max KS 0.19) vs X_test

- `max_ks_distance`: 0.19159928122192277
- `worst_feature_index`: 2.0
- `mean_ks_distance`: 0.1311489218328841
- `warn_ks`: 0.1
- `fail_ks`: 0.2

- train vs X_val: max KS 0.189 at feature 2
- train vs X_test: max KS 0.192 at feature 2

## error.pointwise - PASS
RMSE 0.001451, skill 0.99

- `rmse`: 0.0014506988575394902
- `mae`: 0.000875226366152772
- `max_abs_error`: 0.006132163429155124
- `r2`: 0.9999797155918597
- `skill_vs_mean_baseline`: 0.9933171661386588

Per-channel RMSE:
- channel 0: 0.001763
- channel 1: 8.209e-08
- channel 2: 0.001791

## error.field - SKIP
skipped: output is not a spatial field (field size 3)

## calibration.coverage - SKIP
skipped: no predicted uncertainty (return (mean, std) or set metadata['predicted_std'])

## constraints.physical - PASS
all declared physical constraints satisfied

- `species_nonneg_violation_fraction`: 0.0
- `mass_conservation_violation_fraction`: 0.0
- `max_violation_fraction`: 0.0

- species_nonneg: 0/2862 violations (fraction 0, max magnitude 9.76e-08)
- mass_conservation: 0/954 violations (fraction 0, max magnitude 0.0023)

## export.roundtrip - PASS
exported model matches in-memory model (max abs diff 6.41e-08)

- `max_abs_difference`: 6.405496588701709e-08
- `max_rel_difference`: 2.2299143803695513e-06
- `fraction_mismatched`: 0.0
- `atol`: 0.0001
- `rtol`: 0.001

## speed.inference - PASS
within budget: 1784011 samples/s, 5.61e-07s/sample

- `median_batch_seconds`: 0.0005347499973140657
- `throughput_samples_per_s`: 1784011.2291570583
- `latency_per_sample_s`: 5.605345883795238e-07
- `n_samples`: 954.0
