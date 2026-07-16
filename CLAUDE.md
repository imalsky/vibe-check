# vibe-check - working agreement

Standing instructions for any Claude Code session in this repo. GOAL.md holds the
full mission and is the prompt to paste when kicking off a build session. This
file is the short version that should hold in every session.

## What this project is

vibe-check runs reliability checks on ML surrogates (emulators, neural operators,
learned solver components) before they are trusted inside scientific software. It
builds the validation layer, not new scientific models. Funded by a 2026 URSSI
fellowship; docs/proposal/URSSI.pdf is the source of truth for scope.

## Each session

1. Read ROADMAP.md. Take the next unchecked item in the earliest incomplete
   milestone. Do not skip ahead.
2. Read docs/VALIDATION_CONTRACT.md for the interface every check follows.
3. Do one item end to end (implement, test, document), then commit and push.

## "One check, done"

- Implemented in the right module under src/vibecheck/checks/, accepting the
  orchestrator context and returning a CheckResult. Use Status.SKIP when required
  inputs are missing; never silently pass.
- Registered in vibecheck.core._REGISTERED_CHECKS.
- Covered by a test that constructs both a case it should FAIL and one it should
  PASS. A check with no failing-case test is not done.
- Docstring states what it assumes and what WARN vs FAIL mean.
- ruff check . and pytest both pass before you commit.

## Operating loop

- Small increments: one check or one example per commit. Keep main green.
- Update the relevant ROADMAP.md checkbox in the same commit as the work.
- Commit messages are plain and specific. Do NOT add a Claude / AI co-author
  trailer; commits are Isaac's alone.
- numpy-only core. Plotting lives behind the viz extra. Framework code (torch,
  onnx, jax) is optional and imported lazily inside the check that needs it,
  never at package import time.
- Any number, plot, or report must come from code actually run this session.
  Never invent metrics. If you could not run it, say so and mark it not done.

## Guardrails

- The repo is private. Do not change visibility, publish to PyPI, or cut a public
  release without explicit sign-off from Isaac.
- Do not commit data, trained weights, or large generated artifacts. Scripts
  regenerate them. Nothing under code-for-science-and-society-invoice/ is ever
  committed (financial/personal; already gitignored).
- Ask before a hard-to-reverse API decision the contract does not settle (public
  function names, the shape of metadata). Otherwise proceed and note the choice
  in the commit.

## Writing style

- Plain and direct. No emojis anywhere. No em dashes in prose; use commas,
  colons, or separate sentences. Do not inflate what a check tests.
