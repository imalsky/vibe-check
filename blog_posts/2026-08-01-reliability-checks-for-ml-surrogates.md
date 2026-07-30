---
title: "Reliability checks for machine-learning surrogates in scientific software"
date: 2026-08-01
author: "Isaac Malsky"
---

Machine-learning surrogate models are now common across science. An emulator, neural operator, or learned solver component can reproduce what a classical physical solver computes, at a fraction of the cost. For example, weather and climate models now use learned components for global prediction. These methods make new things possible: higher-resolution models, and uncertainty studies at a scale nobody could afford before.

But the way we decide whether to trust these models has not kept up. Teams usually check three things: a held-out test error, a speedup number, and a handful of example plots. That is not enough. A model with low average error can still fail in specific regions of parameter space. It can silently extrapolate beyond what it was trained on. It can violate physical constraints that the original solver respected by construction. Once a surrogate is built into a larger code, everyone downstream inherits its failures, even people who never trained it. As science relies more on these tools, trust in their accuracy matters more.

Here is one common failure that is hard to see: normalization leakage. I built a small regression surrogate to show it. One input feature is nearly constant in the training data (standard deviation 0.03), but varies normally at test time. This is not a contrived setup. It describes any control knob that was held roughly fixed while training data was collected, but moves freely once the model is deployed. I trained the same model twice and changed exactly one thing: whether the input scaler was fit on the training split alone, or on the full dataset.

The honest pipeline reports a test RMSE of 0.87. The leaky one reports 0.25. Here is why. Fit on training data alone, the scaler divides the test values of that feature by 0.03. That inflates them to a scale of roughly thirty, far outside anything the model saw during training. Fit on everything, it divides by 0.59 instead, because the test set's own spread had already pushed the standard deviation up. So the inputs stay in range, and the model looks accurate. The flattering number cannot be reproduced by anyone who re-runs the workflow honestly. And it hides a real problem: the surrogate is not ready for inputs where that feature varies. This failure mode is not rare. A survey of machine-learning-based science found leakage errors in 17 fields, affecting 329 papers, sometimes leading to badly overoptimistic conclusions (Kapoor and Narayanan 2023). Fitting the scaler before the split, the error above, is one of eight leakage types they catalog.

Right now, there is no standard, practical way to test any of this before a surrogate goes into a larger scientific software system. General-purpose ML testing libraries exist. So do verification and uncertainty-quantification toolkits for simulation. But none of them answer the questions that come up once a neural emulator is part of a physics code: Is the scaler honest? Is the model being asked to extrapolate? Does it still conserve what the solver conserved? Does the exported copy agree with the model that was actually trained?

I have written some version of these checks for every emulator I have built, and thrown the code away each time. That is the itch behind this project.

Over this URSSI fellowship I will:

1. Build an open-source Python package that tests normalization, data-split hygiene, training-domain coverage, constraint violations, calibration, exported inference, and speed.
2. Validate it against representative local, sequential, and spatial-field emulators.
3. Open-source the package, documentation, and tutorial material for the community to use.

The package is called vibe-check. You hand it a predict function and your data splits, numpy in and numpy out. You get back a report that someone who did not train the model can actually read.

This work was supported by the US Research Software Sustainability Institute (URSSI) via grant G-2022-19347 from the Sloan Foundation.
