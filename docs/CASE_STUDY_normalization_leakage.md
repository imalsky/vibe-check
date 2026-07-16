# Case study: how normalization leakage inflates a surrogate's apparent accuracy

This is the URSSI case study deliverable: a short, reproducible look at one
common failure mode in scientific machine learning, and how a reliability check
catches it. The failure mode is normalization leakage, called out in the
reproducibility analysis of Kapoor and Narayanan (2023) as one of the most
frequent causes of inflated results across scientific ML.

## The failure mode

Surrogates almost always normalize their inputs, typically by subtracting a
per-feature mean and dividing by a per-feature standard deviation. The rule is
that those statistics must be estimated from the training split only. If they
are estimated from the full dataset, train plus test, then information about the
held-out data has leaked into the pipeline before evaluation. The reported test
error is then computed with a scaler that has already seen the test set, and it
is no longer a number you can reproduce in deployment, where only training
statistics are available.

The leak is easy to introduce by accident: fit a `StandardScaler` on all the
data, then split. It is also easy to miss, because the model still trains, the
plots still look reasonable, and the headline error can even improve.

## A concrete demonstration

The script `case_study/leakage_demo.py` sets up a small regression surrogate
where one input feature, `x0`, is nearly constant in the training data (standard
deviation 0.03) but varies normally at test time (standard deviation 1.0). This
is a realistic situation: a control knob that was held almost fixed while
collecting training data but moves freely in deployment. The target does not
even depend on `x0`; the feature only matters through how it is scaled.

We fit the same model twice, changing only how the input scaler is estimated:

- **honest**: scaler fit on the training split only (what deployment allows);
- **leaked**: scaler fit on the full dataset, train plus test.

The result, from a fixed seed:

```
train per-feature std: [0.03  0.996 0.99 ]
full  per-feature std: [0.588 1.008 0.986]
honest (train-only scaler) test RMSE: 0.8674
leaked (full-data scaler)  test RMSE: 0.2478
leaked appears 3.5x more accurate
honest: leakage.normalization -> PASS
leaked: leakage.normalization -> FAIL
```

The leaked pipeline looks 3.5 times more accurate. The mechanism is entirely in
the scaler. The train-only standard deviation for `x0` is 0.03, so at test time
the honest scaler divides the varying test values of `x0` by 0.03 and blows them
up to a scale of roughly thirty, far outside anything the model saw in training.
The leaked scaler divides by 0.59 instead, because the full-data standard
deviation was inflated by the test set's own spread, so the test inputs stay in
a sane range and the model looks accurate.

## Why the leaked number is not trustworthy

The 0.25 RMSE is not reproducible. It was produced by a scaler that used the test
data to set its scale. In deployment you do not have the test set; you have the
training statistics, which give the 0.87 RMSE. The flattering number cannot be
recreated by anyone re-running the workflow honestly, and it hides a genuine
problem: this surrogate is not ready for inputs where `x0` varies. Aggregate test
error, the usual headline, reported the model as good precisely because the leak
masked the failure.

Note that the direction of the effect is not the point. In other setups leakage
can leave the number roughly unchanged, or even make it slightly worse. What is
universal is that the reported metric was computed with knowledge of the held-out
data, so it does not describe the model you would actually deploy.

## How vibe-check catches it

The `leakage.normalization` check does not need to know the failure in advance.
The user passes the statistics they actually applied, and the check recomputes
the train-only statistics and the full-data statistics and asks which the
provided ones match. Here it passes the honest pipeline and fails the leaked one,
with the summary "normalization statistics match full-data statistics, not
train-only: preprocessing leaked held-out data." The check runs in the same
report as the accuracy checks, so the leak shows up right next to the flattering
error number rather than being discovered months later.

## Takeaways

- Fit every preprocessing step, scalers included, on the training split only,
  then apply it to validation and test.
- A low aggregate test error is necessary but not sufficient. Report it next to
  a leakage check, not on its own.
- Watch features whose training variance is small. Standardizing by a tiny
  standard deviation is unstable, and it is exactly where leaked and honest
  scalers diverge the most.

## Reproduce

```
pip install -e ".[viz]"
pip install numpy scikit-learn
python docs/case_study/leakage_demo.py
```

## Reference

Kapoor, S. and Narayanan, A. Leakage and the reproducibility crisis in
machine-learning-based science. Patterns, 4, 100804, 2023.
