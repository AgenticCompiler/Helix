# AGENTS.md

## Project Overview

- This repository provides a small `uv`-managed CLI for Triton Ascend NPU operator workflows.
- The CLI is a wrapper around code agents plus local skills, not a replacement for the skills themselves.

## Core Principles

- Keep prompts, comments, logs, and user-visible instructions in English.
- Do not create or switch to a different git branch unless the user has explicitly confirmed that branch change.
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
- When modifying `skills/common/ascend-npu-optimize-submit-baseline/references/contract.json` or `skills/common/ascend-npu-optimize-submit-round/references/contract.json`, rerun `python3 skills/triton/triton-npu-optimize/script/update-artifacts.py` so `skills/triton/triton-npu-optimize/references/artifacts.md` stays in sync.
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
