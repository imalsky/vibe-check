# Goal: build `vibe-check`

Paste the section below into Claude Code goal mode (or run it as a standing
autonomous goal). It is written to be self-contained; everything it references
lives in this repository.

---

## Mission

Build `vibe-check`, an open-source Python package that runs reliability checks
on machine-learning surrogates (emulators, neural operators, learned solver
components) before they are trusted inside a scientific software stack. The
package turns common failure modes of scientific ML into clear validation
reports a non-author can read. This is funded by a 2026 URSSI fellowship; the
proposal is at `docs/proposal/URSSI.pdf` and is the source of truth for scope.

Deliver, over the fellowship: the package, three worked example reports
(Robertson, Lorenz, FNO), a reusable workflow, a tutorial, and a short case
study on normalization leakage.

## Where to start each session

1. Read `ROADMAP.md`. Pick the next unchecked item in the earliest incomplete
   milestone. Do not skip ahead.
2. Read `docs/VALIDATION_CONTRACT.md` for the interface every check follows.
3. Do one item end to end (implement, test, document), then commit and push.

## What "one check, done" means

- Implemented in the correct module under `src/vibecheck/checks/`, accepting the
  orchestrator context and returning a `CheckResult`. Use `Status.SKIP` when the
  inputs a check needs were not provided; never silently pass.
- Registered in `vibecheck.core._REGISTERED_CHECKS`.
- Covered by a real test in `tests/` that constructs a case the check should
  FAIL and a case it should PASS. A check with no failing-case test is not done.
- Documented: what it assumes, and what WARN vs FAIL mean, in its docstring.
- `ruff check .` and `pytest` both pass locally before you commit.

## Operating loop (how to work in goal mode)

- Work in small increments. One check or one example per commit. Keep `main`
  green; if CI would fail, fix it before moving on.
- After each working increment: update the relevant `ROADMAP.md` checkbox in the
  same commit, then `git commit` and `git push`. Commit messages are plain and
  specific ("implement leakage.normalization check + tests"). Do not add any
  Claude / AI co-author trailer; commits are Isaac's.
- Prefer a numpy-only core. Plotting goes behind the `viz` extra. Anything
  framework-specific (torch, onnx, jax) is an optional dependency, imported
  lazily inside the check that needs it, never at package import time.
- When you produce numbers, plots, or example reports, they must come from code
  you actually ran in this session. Never invent metrics or paste a plausible
  result. If you could not run something, say so and mark it not done.

## Definition of done (project level)

- All checks in `docs/VALIDATION_CONTRACT.md` implemented, registered, tested.
- `report.to_markdown` and `report.to_html` both render a real multi-check run.
- The three examples each train a small surrogate, run `vibe-check`, and commit
  a report template. Examples run from a clean checkout with documented commands.
- Tutorial / workflow doc walks a new user from install to a report.
- CI green across the supported Python versions.

## Guardrails and non-goals

- This project builds the validation layer, not new scientific models. Keep the
  example surrogates small and standard.
- Keep the package lightweight and dependency-light. If a check needs a heavy
  dependency, make it optional and degrade to `SKIP` when it is absent.
- Do not commit data, trained weights, or large generated artifacts (they are
  gitignored). Scripts regenerate them.
- The repo is currently private. Do not change its visibility, publish to PyPI,
  or cut a public release without explicit sign-off from Isaac.
- Ask before making a hard-to-reverse API decision that the contract does not
  already settle (public function names, the shape of `metadata`). Otherwise
  proceed and note the choice in the commit.

## Writing style (applies to all prose, docs, comments, reports)

- Plain and direct. No emojis anywhere. No em dashes in prose; use commas,
  colons, or separate sentences.
- Do not inflate. Describe what a check actually tests, not what it aspires to.

## Stop / check-in points

Pause and summarize for Isaac at each milestone boundary in `ROADMAP.md` (end of
M1, M2, M3, M4), and any time a decision needs his input. Otherwise keep going.
