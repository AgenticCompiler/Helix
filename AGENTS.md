# AGENTS.md

## Project Overview

- This repository provides a small `uv`-managed CLI for Triton Ascend NPU operator workflows.
- The CLI is a wrapper around code agents plus local skills, not a replacement for the skills themselves.
- The supported backends are `codex`, `opencode`, `pi`, and `claude`.

## Core Principles

- Keep prompts, comments, logs, and user-visible instructions in English.
- Treat the local `skills/` directory as the source of truth for workflow behavior.
- Write skills as natural-language task guides first; CLI flags are wrapper-specific context, not the primary workflow interface.
- Treat the public operator entrypoint as the API surface for generation workflows.
- When a skill needs to invoke project commands, prefer bundled helper scripts over assuming installed console entrypoints.
- Keep the CLI thin: orchestration belongs in the CLI, while evaluation and workflow logic should remain in the skills unless the CLI truly needs it.
- Preserve a clear separation between generic agent flow and backend-specific details.
- Prefer additive diagnostics that improve visibility without changing command semantics.
- Default to protecting existing generated artifacts and require explicit overwrite behavior to replace them.
- Keep mode selection scoped to the commands that own it, use explicit defaults for generation and optimize flows, and prefer reusing generated metadata when continuing or executing existing harnesses.
- For optimize flows, treat resume behavior explicitly: `auto` may continue only from a complete existing optimize session, `continue` must fail fast when required session artifacts are missing, and `fresh` must refuse to run when optimize artifacts already exist.
- For optimize flows, prefer reusing existing test and benchmark harnesses; generate them only when the workspace is missing required validation artifacts.
- For optimize flows, require every optimization round to record why the chosen change may help and what evidence supports it.
- For optimize flows, do not default to blind tiling or launch-parameter search when the available evidence does not justify that direction.
- Prefer short actionable CLI validation errors over Python tracebacks.
- Prefer explicit failures over silent fallbacks when expected artifacts or metadata are missing.

## Workspace And Skills

- Before launching a code agent, stage this repository's `skills/` directory into the target workspace in the backend-native location.
- Stage skills by copying content into the workspace instead of creating symlinks.
- If a target skill path already exists as a symlink, fail explicitly instead of reusing it.
- Clean up only the copied skill paths created by the current run.
- Never delete or replace user-owned files or directories during cleanup.
- Treat the top-level `workspace/` directory as a placeholder area for local experimentation, not as repository-owned source, fixture, or verification input.

## Agent Backends

- New backends should follow the same high-level lifecycle: prepare workspace, launch agent, collect result, and clean up.
- Backend-specific command construction should stay isolated from CLI parsing and prompt construction.
- Interactive mode should attach to the live agent UI or session.
- Non-interactive mode should be script-friendly and return a meaningful process exit code.
- PTY-backed non-interactive streaming should treat platform-specific PTY EOF during normal child exit as clean shutdown while still surfacing real read failures.
- Backend-specific launch flags and invocation details belong in `README.md` or focused docs, not here.

## Verification

- Use `uv run --group dev ruff check` for lint checks.
- Use `uv run pyright` for static type checks.
- Use `uv run python -m unittest discover -s tests -v` for the current test suite.

## Design And Documentation Style

- Write a short design document before implementing behavior changes.
- Keep design/spec documents under `docs/specs/`.
- Keep implementation plans under `docs/plans/`.
- Keep behavior and workflow documents under `docs/` with date-prefixed filenames such as `YYYY-MM-DD-<topic>.md`.
- Update `AGENTS.md` when durable project rules change; keep implementation detail in `README.md` and focused docs.
- Document behavior in terms of user-visible semantics first and implementation details second.
- Use `AGENTS.md` for stable project rules and workflow expectations.

## Scope Guardrails

- Do not move implementation detail from skills into the CLI unless the CLI truly needs it for orchestration.
- Keep archived IR capture and inspection flows in `skills/ascend-operator-ir-analyzer/scripts/`; use `capture_ir.py --ir-dir ...` for collection and `inspect_ir.py --ir-dir ...` for stage navigation, summaries, and diffs instead of growing new CLI subcommands.
- Do not couple the project to a single operator style beyond what the existing skills already assume.
- Keep documentation at the overview and workflow level unless a file explicitly needs lower-level implementation detail.
