"""Individual diagnostic checks.

Each module here implements one family of checks from the validation contract
(``docs/VALIDATION_CONTRACT.md``). A check is a function that accepts the
orchestrator context (``predict`` plus the optional data splits and
``metadata``) and returns a :class:`vibecheck.core.CheckResult`.

All ten checks are implemented and registered in
``vibecheck.core._REGISTERED_CHECKS``: leakage (normalization, split_overlap),
distribution (coverage, drift), error (pointwise, field), calibration
(coverage), constraints (physical), export (roundtrip), and speed (inference).
Each returns ``Status.SKIP`` when its required inputs are missing rather than
passing silently, and reads its thresholds from ``metadata``.
"""
