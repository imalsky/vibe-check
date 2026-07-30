# Case study: how normalization leakage inflates a surrogate's apparent accuracy

This is the URSSI case study deliverable: a short, reproducible look at one
common failure mode in scientific machine learning, and how a reliability check
catches it. The failure mode is normalization leakage. Kapoor and Narayanan
(2023) call it out in their reproducibility analysis as one of the most
frequent causes of inflated results across scientific ML.

## The failure mode

Surrogates almost always normalize their inputs. This usually means
subtracting a per-feature mean and dividing by a per-feature standard
deviation. The rule is simple: fit those statistics on the training split
only. If you fit them on the full dataset (train plus test), information
about the held-out data leaks into the pipeline before evaluation. The
reported test error is then computed with a scaler that has already seen the
test set. That number is not reproducible in deployment, where only training
statistics are available.

This leak is easy to introduce by accident: fit a `StandardScaler` on all the
data, then split. It is also easy to miss. The model still trains. The plots
still look reasonable. The headline error can even improve.

## A concrete demonstration

The script `case_study/leakage_demo.py` sets up a small regression surrogate.
One input feature, `x0`, is nearly constant in the training data (standard
deviation 0.03), but varies normally at test time (standard deviation 1.0).
This is a realistic situation: a control knob that was held almost fixed
while collecting training data, but moves freely in deployment. The target
does not even depend on `x0`; the feature only matters through how it is
scaled.

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

The leaked pipeline looks 3.5 times more accurate. The mechanism is entirely
in the scaler. The train-only standard deviation for `x0` is 0.03. So at test
time, the honest scaler divides the varying test values of `x0` by 0.03. That
blows them up to a scale of roughly thirty, far outside anything the model
saw in training. The leaked scaler divides by 0.59 instead, because the
full-data standard deviation was inflated by the test set's own spread. So
the test inputs stay in a normal range, and the model looks accurate.

## Why the leaked number is not trustworthy

The 0.25 RMSE is not reproducible. A scaler that used the test data to set
its scale produced it. In deployment you do not have the test set; you have
the training statistics, which give the 0.87 RMSE. Nobody re-running the
workflow honestly can recreate the flattering number. It also hides a real
problem: this surrogate is not ready for inputs where `x0` varies. The usual
headline number, aggregate test error, called the model good exactly because
the leak hid the failure.

The direction of the effect is not the point here. In other setups, leakage
can leave the number roughly unchanged, or even make it slightly worse. What
stays true every time: the reported metric was computed with knowledge of the
held-out data. So it does not describe the model you would actually deploy.

## How vibe-check catches it

The `leakage.normalization` check does not need to know the failure in
advance. You pass it the statistics you actually applied. The check
recomputes the train-only statistics and the full-data statistics, and asks
which set the provided ones match. Here, it passes the honest pipeline and
fails the leaked one, with the summary "normalization statistics match
full-data statistics, not train-only: preprocessing leaked held-out data."
The check runs in the same report as the accuracy checks. So the leak shows
up right next to the flattering error number, instead of being discovered
months later.

## Takeaways

- Fit every preprocessing step, scalers included, on the training split only,
  then apply it to validation and test.
- A low aggregate test error is necessary but not sufficient. Report it next to
  a leakage check, not on its own.
- Watch features with small training variance. Standardizing by a tiny
  standard deviation is unstable. It is exactly where leaked and honest
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
