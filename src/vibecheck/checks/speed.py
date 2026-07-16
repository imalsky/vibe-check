"""Speed checks: is inference fast enough for its intended use.

The entire point of a surrogate is speed. This check measures inference
throughput and latency (with warmup, batched and unbatched) and compares them
against a user-supplied budget, so "fast enough" is stated rather than assumed.
"""

from __future__ import annotations

from typing import Any

from ..core import CheckResult


def inference(**context: Any) -> CheckResult:
    """Measure inference throughput / latency against a user-set budget."""
    raise NotImplementedError


inference.check_name = "speed.inference"
