# Contributing

`vibe-check` is shaped by community feedback. Different scientific domains
need different reliability checks. The most useful contribution right now is
telling us which checks are missing, too narrow, or too expensive for routine
use. Open an issue; there is a "Diagnostic request or feedback" template.

## Development

```
pip install -e ".[dev,viz]"
ruff check .
pytest
```

## Adding a check

1. Implement it in the right module under `src/vibecheck/checks/`. Follow the
   contract in `docs/VALIDATION_CONTRACT.md`: accept the orchestrator context,
   return a `CheckResult`, and use `SKIP` when required inputs are missing.
2. Register it in `vibecheck.core._REGISTERED_CHECKS`.
3. Add a test. A check must never silently pass.
4. Document what it assumes and what WARN vs FAIL means.

Keep the core dependency footprint small (numpy only). Plotting lives behind
the `viz` extra. Framework-specific code (torch, onnx) stays optional.
