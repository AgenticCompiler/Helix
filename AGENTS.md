# AGENTS.md

## Project Overview

- This repository provides a small `uv`-managed CLI for Triton Ascend NPU operator workflows.
- The CLI is a wrapper around code agents plus local skills, not a replacement for the skills themselves.
- The current supported backends are `codex` and `opencode`.

## User-Facing Commands

- `gen-test`: generate correctness tests for an operator file
- `run-test`: execute generated correctness tests for an operator file
- `gen-bench`: generate performance benchmarks for an operator file
- `run-bench`: execute generated benchmarks for an operator file
- `optimize`: optimize an operator file with long-running supervision
- The CLI may accept compatibility aliases such as snake_case spellings, but kebab-case remains the canonical displayed command form.
- `run-test` should require both `--test-file` and `--operator-file`.
- `run-bench` should require both `--bench-file` and `--operator-file`.

## Core Principles

- Keep prompts, comments, logs, and user-visible instructions in English.
- Treat the local `skills/` directory as the source of truth for workflow behavior.
- Write skills as natural-language task guides first; treat CLI flags as wrapper-specific context rather than the primary skill interface.
- When a skill needs to invoke project commands, prefer a bundled script under `skills/scripts/` over assuming an installed console entrypoint.
- When a skill depends on a bundled helper script, include a few short command templates instead of only mentioning the script abstractly.
- Keep the CLI thin: it should orchestrate agent execution, not reimplement skill logic.
- Preserve a clear separation between generic agent flow and backend-specific details.
- Prefer optional diagnostic flags for orchestration visibility instead of always-on debug output.
- When adding orchestration flags, keep them additive: they may increase visibility, but should not change the underlying agent task semantics.
- When verbose diagnostics cover workspace preparation, include both setup and cleanup visibility so link lifecycle is auditable.
- Make verbose output readable first: short categories, visible link targets, and separate command/prompt display beat raw shell dumps.
- When a command writes generated artifacts, default to protecting existing files and require an explicit overwrite flag to replace them.
- If overwrite is explicitly requested for a generated artifact, remove the old file in the CLI layer before launching the agent.
- Keep command-specific mode flags scoped narrowly; for example, test-mode selection belongs only to test generation and test execution.
- Default command-specific modes to an explicit value instead of leaving them implicit; use `standalone` for test and benchmark flows unless the user asks for another mode.
- For `optimize`, default to `differential` test validation and `standalone` benchmark validation unless the user asks for another combination.
- Likewise, benchmark-mode selection belongs only to benchmark generation and benchmark execution.
- For expected CLI validation failures, prefer short actionable error messages over Python tracebacks.
- Prefer explicit failures over silent fallbacks when an expected file or artifact is missing.

## Workspace and Skill Handling

- Before launching a code agent, expose this repository's `skills/` directory inside the target workspace in the backend-specific location.
- For Codex, use `.codex/skills`.
- For OpenCode, use `.opencode/skills/<name>/SKILL.md` via per-skill directory links.
- If an existing skill symlink already points to this repository's source skill directory, treat it as satisfied and skip recreating it.
- Clean up only the symlinks created by the current run.
- Never delete or replace user-owned files or directories during cleanup.

## Agent Backend Expectations

- New backends should follow the same high-level lifecycle: prepare workspace, launch agent, collect result, clean up.
- Backend-specific command construction should stay isolated from CLI parsing and prompt construction.
- Interactive mode should attach to the live agent UI or session.
- Non-interactive mode should be script-friendly and return a meaningful process exit code.
- PTY-backed non-interactive streaming should treat platform-specific PTY EOF during normal child exit as clean shutdown, while still surfacing real read failures.
- The Codex backend should launch both interactive and non-interactive runs with `--ephemeral` and `--skip-git-repo-check`.

## Optimize Command Expectations

- `optimize` is treated as a long-running workflow.
- Supervision should detect stalls conservatively and attempt recovery without hiding failures.
- Automatic recovery should prefer continuing from recent progress before starting over.
- The `optimize` skill should search over validated candidate branches, not assume every new round must continue from the current best version.
- The `optimize` knowledge base should offer a compact pattern index first and only then drill into one or two detailed optimization pattern references.
- The `optimize` command may install a temporary workspace `AGENTS.md` with run-specific guardrails; if the workspace already has one, back it up first and restore it after the run.

## Verification

- Use `uv run --group dev ruff check` for lint checks.
- Use `uv run pyright` for static type checks.
- Use `uv run python -m unittest discover -s tests -v` for the current test suite.

## Design And Documentation Style

- Write a short design document before implementing behavior changes.
- Keep design and behavior documents under `docs/` with date-prefixed filenames such as `YYYY-MM-DD-<topic>.md`.
- When behavior changes, update the corresponding design doc, `README.md`, tests, and `AGENTS.md` together.
- Document behavior in terms of user-visible semantics first, then implementation details second.
- Use `AGENTS.md` for durable project rules and workflow expectations.
- Use `docs/` for detailed behavior descriptions and per-change design decisions.

## Scope Guardrails

- Do not move implementation detail from skills into the CLI unless the CLI truly needs it for orchestration.
- Do not couple the project to a single operator style beyond what the existing skills already assume.
- Keep documentation at the overview and workflow level unless a file explicitly needs lower-level implementation detail.
