# vibe-check report: Robertson state-to-state MLP

Overall: **WARN**

| check | status | summary |
| --- | --- | --- |
| leakage.split_overlap | WARN | 1401 near-duplicate row(s) across splits (standardized distance < 0.001) |
| distribution.drift | WARN | moderate marginal drift (max KS 0.19) vs X_test |
| leakage.normalization | PASS | normalization statistics match the train-only statistics |
| distribution.coverage | PASS | 0.0% of test points outside the training domain (within tolerance) |
| error.pointwise | PASS | RMSE 0.001451, skill 0.99 |
| constraints.physical | PASS | all declared physical constraints satisfied |
| export.roundtrip | PASS | exported model matches in-memory model (max abs diff 6.41e-08) |
| speed.inference | PASS | within budget: 1782761 samples/s, 5.61e-07s/sample |
| error.field | SKIP | skipped: output is not a spatial field (field size 3) |
| calibration.coverage | SKIP | skipped: no predicted uncertainty (return (mean, std) or set metadata['predicted_std']) |

## leakage.normalization - PASS
normalization statistics match the train-only statistics

- `rel_distance_to_train_stats`: 0
- `rel_distance_to_full_stats`: 0.02654
- `rtol`: 0.001

## leakage.split_overlap - WARN
1401 near-duplicate row(s) across splits (standardized distance < 0.001)

- `exact_duplicate_rows`: 0
- `near_duplicate_rows`: 1401
- `min_standardized_distance`: 9.307e-08
- `atol`: 0.001

- X_train vs X_val: 0 exact, 603 near-duplicate rows
- X_train vs X_test: 0 exact, 617 near-duplicate rows
- X_val vs X_test: 0 exact, 181 near-duplicate rows
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
moderate marginal drift (max KS 0.19) vs X_test

- `max_ks_distance`: 0.1916
- `worst_feature_index`: 2
- `mean_ks_distance`: 0.1311
- `warn_ks`: 0.1
- `fail_ks`: 0.2

- train vs X_val: max KS 0.189 at feature 2
- train vs X_test: max KS 0.192 at feature 2

(2 figures attached; see the HTML report)

## error.pointwise - PASS
RMSE 0.001451, skill 0.99

- `rmse`: 0.001451
- `mae`: 0.0008752
- `max_abs_error`: 0.006132
- `r2`: 1
- `skill_vs_mean_baseline`: 0.9933
- `warn_skill`: 0.5

Per-channel RMSE:
- channel 0: 0.001763
- channel 1: 8.209e-08
- channel 2: 0.001791

(2 figures attached; see the HTML report)

## error.field - SKIP
skipped: output is not a spatial field (field size 3)

## calibration.coverage - SKIP
skipped: no predicted uncertainty (return (mean, std) or set metadata['predicted_std'])

## constraints.physical - PASS
all declared physical constraints satisfied

- `constraint_species_nonneg_violation_fraction`: 0
- `constraint_mass_conservation_violation_fraction`: 0
- `max_violation_fraction`: 0

- species_nonneg: 0/2862 violations (fraction 0, max magnitude 9.76e-08)
- mass_conservation: 0/954 violations (fraction 0, max magnitude 0.0023)

## export.roundtrip - PASS
exported model matches in-memory model (max abs diff 6.41e-08)

- `max_abs_difference`: 6.405e-08
- `max_rel_difference`: 2.23e-06
- `fraction_mismatched`: 0
- `atol`: 0.0001
- `rtol`: 0.001

## speed.inference - PASS
within budget: 1782761 samples/s, 5.61e-07s/sample

- `median_batch_seconds`: 0.0005351
- `throughput_samples_per_s`: 1.783e+06
- `latency_per_sample_s`: 5.609e-07
- `n_samples`: 954
- `min_throughput`: 10000
