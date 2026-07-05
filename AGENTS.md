# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Read Before You Write

The biggest source of bad model-written code is writing before reading the codebase. Read the files you are about to touch; read, not skim. Copy the patterns that already exist, and check the imports to see what the project actually depends on, so you do not reach for axios where everything is fetch. When you cannot find a pattern, ask instead of guessing.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Verification

The gap between code that works and code you think works is testing. When fixing a bug, write the failing test first, watch it fail, then fix it; that is the only proof you fixed the cause and not the symptom. Test behavior that can actually break, not that a constructor sets a field. If something is hard to test, that is information about the design, not permission to skip it.

## 6. Debugging

When something breaks, investigate; do not guess. Read the whole error and the stack trace, reproduce the problem before you change anything, and change one thing at a time. Do not paper over an unexpected null with a null check; find out why it is null, or the bug just moves somewhere quieter.

## 7. Dependencies

Every dependency is permanent code you do not control. Before adding one, ask whether the project or the standard library can already do it with crypto.randomUUID() over a uuid package. When you do add one, say why, so the choice is visible rather than smuggled into the manifest.

## 8. Communication

Say what you did and why, not just a block of code. Flag concerns even when you did exactly what was asked, and be precise about uncertainty: "I am not sure this library supports streaming" tells the user what to verify; "I think this should work" does not.

## 9. Common Failure Modes

A few patterns recur often enough to name: the Kitchen Sink (restructuring half the codebase while you are at it), the Wrong Abstraction (copy-paste twice before you abstract), the Optimistic Path (the happy path handled and the 500 ignored), and the Runaway Refactor (a fix that cascades across files). Catch yourself in any of these and the right move is to stop, not to push through.

---

## Project Overview

- This repository provides a small `uv`-managed CLI for Triton Ascend NPU operator workflows.
- The CLI is a wrapper around code agents plus local skills, not a replacement for the skills themselves.

## Core Principles

- Keep prompts, comments, logs, and user-visible instructions in English.
- Do not create or switch to a different git branch unless the user has explicitly confirmed that branch change.
- When creating a new branch after explicit confirmation, use an appropriate type prefix such as `feat/`, `bugfix/`, `refact/`, `docs/`, or `chore/`.
- Treat the local `skills/` directory as the source of truth for workflow behavior, and write skills as natural-language task guides first.
- When a skill needs to invoke project commands, prefer bundled helper scripts over assuming installed console entrypoints.
- Keep the CLI thin: orchestration belongs in the CLI, while evaluation and workflow logic stay in skills unless the CLI truly needs them.
- Treat this repository as an executable application first, not a reusable third-party library.
- Preserve clear boundaries between generic agent flow, backend-specific behavior, and feature-local implementation.
- Prefer names, module boundaries, and contracts that match real ownership. Rename, move, or delete redundant layers instead of preserving misleading abstractions or compatibility shims.
- Prefer feature-local modules and data over top-level shared helpers unless multiple subsystems truly share the behavior.
- Prefer additive diagnostics, short actionable validation errors, explicit failures, and protected generated artifacts over silent fallback or implicit overwrite.

## Command-Line And Skills

- This CLI implemented in this repository is a thin wrapper around code agents plus local skills. So, before launching a code agent, stage this repository's `skills/` directory into the target workspace in the backend-native location. Therefore, when adding or removing a repository skill, or when adding a new CLI subcommand that stages skills, review and update `src/triton_agent/skill_staging.py` so the centralized staging table stays in sync.
- Keep `skills/*/scripts/` self-contained: skill-side Python helpers must not import `triton_agent`. If runtime code needs to reuse a skill-script implementation, load it through the existing bridge layer in `src/triton_agent/skill_loader.py` instead of creating a reverse dependency from the skill back into `src/`.
- When modifying Python files under `skills/*/scripts/`, always run the additional file-scoped `pyright` strict check via `bash scripts/run-skill-script-pyright.sh skills/path/to/script.py` before considering the change complete, even though the repository default keeps those scripts in basic mode.
- When modifying `skills/common/ascend-npu-optimize-state/references/baseline-contract.json` or `skills/common/ascend-npu-optimize-state/references/round-contract.json`, rerun `python3 skills/triton/triton-npu-optimize/script/update-artifacts.py` so `skills/triton/triton-npu-optimize/references/artifacts.md` stays in sync.
- Follow the shared backend architecture and lifecycle contract documented in `docs/specs/2026-04-13-cli-backend-dedup-refactor-design.md`; keep backend-specific launch flags and invocation details in `README.md` or focused docs, not here.
- Optimization pattern/symptom files should be created and updated using the `create-optimize-pattern` skill.

## Design And Documentation Style

- Write a short design document before implementing behavior changes.
- Keep documents in proper places:
  - design/spec documents under `docs/specs/`
  - implementation plans under `docs/plans/`
  - review and audit reports under `docs/reviews/`.
  - focused behavior and workflow documents under `docs/notes/`

- Documents must be named with date-prefixed filenames such as `YYYY-MM-DD-<topic>.md`.
- Document behavior in terms of user-visible semantics first and implementation details second.

- Update `AGENTS.md` only when durable project rules change; keep implementation detail, command examples, and feature semantics in `README.md` and focused docs.

- Use `AGENTS.md` for stable project rules and workflow expectations.

- Keep human-facing contract prose in skills or focused references, and keep machine-readable contracts in one loadable source rather than duplicating field lists across prompts, checkers, and runtime code.

- Use the standard repository verification commands:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`.

## Scope Guardrails

- Do not move implementation detail from skills into the CLI unless the CLI truly needs it for orchestration.
- Do not couple the project to a single operator style beyond what the existing skills already assume.
