# AGENTS.md

## Project Overview

- This repository provides a small `uv`-managed CLI for Triton Ascend NPU operator workflows.
- The CLI is a wrapper around code agents plus local skills, not a replacement for the skills themselves.
- The supported backends are `codex`, `opencode`, `pi`, and `claude`.

## Core Principles

- Keep prompts, comments, logs, and user-visible instructions in English.
- Treat the local `skills/` directory as the source of truth for workflow behavior, and write skills as natural-language task guides first.
- Treat the public operator entrypoint as the API surface for generation workflows.
- When a skill needs to invoke project commands, prefer bundled helper scripts over assuming installed console entrypoints.
- Keep the CLI thin: orchestration belongs in the CLI, while evaluation and workflow logic stay in skills unless the CLI truly needs them.
- Treat this repository as an executable application first, not a reusable third-party library.
- Preserve clear boundaries between generic agent flow, backend-specific behavior, and feature-local implementation.
- Prefer names, module boundaries, and contracts that match real ownership. Rename, move, or delete redundant layers instead of preserving misleading abstractions or compatibility shims.
- Prefer feature-local modules and data over top-level shared helpers unless multiple subsystems truly share the behavior.
- Prefer additive diagnostics, short actionable validation errors, explicit failures, and protected generated artifacts over silent fallback or implicit overwrite.
- Keep optimize workflows explicit, evidence-driven, and role-separated: resume semantics must be clear, reusable harnesses should be reused, worker and supervisor responsibilities should stay distinct, and each round should record why a change may help.

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

## Design And Documentation Style

- Write a short design document before implementing behavior changes.
- Keep design/spec documents under `docs/specs/`.
- Keep implementation plans under `docs/plans/`.
- Keep focused behavior and workflow documents under `docs/notes/` with date-prefixed filenames such as `YYYY-MM-DD-<topic>.md`.
- Keep review and audit reports under `docs/reviews/`.
- Update `AGENTS.md` only when durable project rules change; keep implementation detail, command examples, and feature semantics in `README.md` and focused docs.
- Document behavior in terms of user-visible semantics first and implementation details second.
- Use `AGENTS.md` for stable project rules and workflow expectations.
- Keep human-facing contract prose in skills or focused references, and keep machine-readable contracts in one loadable source rather than duplicating field lists across prompts, checkers, and runtime code.
- Use the standard repository verification commands documented in `README.md`.

## Scope Guardrails

- Do not move implementation detail from skills into the CLI unless the CLI truly needs it for orchestration.
- Keep archived IR capture and inspection flows in `skills/triton-npu-analyze-ir/scripts/` instead of growing new CLI subcommands for that workflow.
- Do not couple the project to a single operator style beyond what the existing skills already assume.
- Keep documentation at the overview and workflow level unless a file explicitly needs lower-level implementation detail.
