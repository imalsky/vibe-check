"""Speed checks: is inference fast enough for its intended use.

The entire point of a surrogate is speed. This check measures inference
throughput and latency (with warmup) and compares them against a user-supplied
budget, so "fast enough" is stated rather than assumed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .._utils import skip
from ..core import CheckResult, Status


def inference(**context: Any) -> CheckResult:
    """Measure inference throughput / latency against a user-set budget.

    Runs ``predict`` on ``X_test`` a few times after a warmup and reports the
    median batch time, throughput (samples/second), and per-sample latency. The
    budget lives under ``metadata['speed']``:

    - ``min_throughput``: required samples/second (FAIL if slower);
    - ``max_latency_s``: allowed seconds per sample (FAIL if slower);
    - ``warmup`` (default 1) and ``repeats`` (default 3) control the timing.

    With no budget set the check cannot say whether the model is fast enough, so
    it SKIPs, carrying the measured numbers in ``metrics`` for the report rather
    than passing silently. Needs ``predict`` and a non-empty ``X_test``.

    Timing is machine-dependent; treat a single run as indicative, not exact.
    """
    name = "speed.inference"
    metadata = context.get("metadata") or {}
    cfg = metadata.get("speed") or {}
    predict = context.get("predict")
    x_test = context.get("X_test")
    if predict is None or x_test is None:
        return skip(name, "need predict and X_test")

    x = np.asarray(x_test)
    n = int(x.shape[0]) if x.ndim >= 1 else 0
    if n == 0:
        return skip(name, "X_test is empty")

    warmup = int(cfg.get("warmup", 1))
    repeats = max(1, int(cfg.get("repeats", 3)))
    try:
        for _ in range(max(0, warmup)):
            predict(x)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            predict(x)
            times.append(time.perf_counter() - start)
    except Exception as exc:
        return CheckResult(
            name=name, status=Status.FAIL, summary=f"predict raised: {exc!r}"
        )

    median_t = float(np.median(times))
    throughput = n / median_t if median_t > 0 else float("inf")
    latency = median_t / n
    metrics = {
        "median_batch_seconds": median_t,
        "throughput_samples_per_s": throughput,
        "latency_per_sample_s": latency,
        "n_samples": float(n),
    }

    min_throughput = cfg.get("min_throughput")
    max_latency = cfg.get("max_latency_s")
    if min_throughput is None and max_latency is None:
        return CheckResult(
            name=name,
            status=Status.SKIP,
            summary=(
                f"measured {throughput:.0f} samples/s; no budget set "
                "(metadata['speed']) to evaluate against"
            ),
            metrics=metrics,
        )

    failures = []
    if min_throughput is not None and throughput < float(min_throughput):
        failures.append(
            f"throughput {throughput:.0f} < required {float(min_throughput):.0f} samples/s"
        )
    if max_latency is not None and latency > float(max_latency):
        failures.append(
            f"latency {latency:.3g}s > budget {float(max_latency):.3g}s per sample"
        )

    if failures:
        return CheckResult(
            name=name, status=Status.FAIL, summary="; ".join(failures), metrics=metrics
        )
    return CheckResult(
        name=name,
        status=Status.PASS,
        summary=f"within budget: {throughput:.0f} samples/s, {latency:.3g}s/sample",
        metrics=metrics,
    )


inference.check_name = "speed.inference"
