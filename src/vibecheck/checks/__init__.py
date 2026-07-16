"""Individual diagnostic checks.

Each module here implements one family of checks from the validation contract
(``docs/VALIDATION_CONTRACT.md``). A check is a function that accepts the
orchestrator context (``predict`` plus the optional data splits and
``metadata``) and returns a :class:`vibecheck.core.CheckResult`.

These are stubs. Implementing them, wiring them into
``vibecheck.core._REGISTERED_CHECKS``, and covering each with a test is the
work tracked in ``ROADMAP.md``.
"""
