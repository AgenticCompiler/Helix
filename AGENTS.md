# AGENTS.md

## Project Overview

- This repository provides a small `uv`-managed CLI for Triton Ascend NPU operator workflows.
- The CLI is a wrapper around code agents plus local skills, not a replacement for the skills themselves.
- The supported backends are `codex`, `opencode`, `pi`, `claude`, `openhands`, and `traecli`.

## Core Principles

- Keep prompts, comments, logs, and user-visible instructions in English.
- Do not create or switch to a different git branch unless the user has explicitly confirmed that branch change.
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
- Treat `skills/triton-npu-optimize-knowledge/references/patterns/*.md` as the authored source of truth for generic optimize patterns; after changing a pattern card, regenerate and commit the checked-in pattern index instead of hand-editing it.
- Treat `skills/triton-npu-optimize-knowledge/references/symptoms/*.md` as the authored source of truth for generic optimize symptoms; after changing a symptom card, regenerate and commit the checked-in symptom index instead of hand-editing it.

## Workspace And Skills

- Before launching a code agent, stage this repository's `skills/` directory into the target workspace in the backend-native location.
- When adding or removing a repository skill, or when adding a new CLI subcommand that stages skills, review and update `src/triton_agent/skill_staging.py` so the centralized staging table stays in sync.
- Stage skills by copying content into the workspace instead of creating symlinks.
- If a target skill path already exists as a symlink, fail explicitly instead of reusing it.
- Clean up only the copied skill paths created by the current run.
- Never delete or replace user-owned files or directories during cleanup.
- Treat the top-level `workspace/` directory as a placeholder area for local experimentation, not as repository-owned source, fixture, or verification input.
- Keep `skills/*/scripts/` self-contained: skill-side Python helpers must not import `triton_agent`. If runtime code needs to reuse a skill-script implementation, load it through the existing bridge layer in `src/triton_agent/skill_loader.py` instead of creating a reverse dependency from the skill back into `src/`.
- When modifying Python files under `skills/*/scripts/`, always run the additional file-scoped `pyright` strict check via `bash scripts/run-skill-script-pyright.sh skills/path/to/script.py` before considering the change complete, even though the repository default keeps those scripts in basic mode.

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
- Use the standard repository verification commands documented in `README.md`: `uv run --group dev ruff check`, `uv run pyright`, and `uv run python -m unittest discover -s tests -v`.
- When running tests with pytest (e.g. during agent workflows), always pass `-q --tb=short --no-header -p no:warnings` to avoid wasting context on per-test verbose output or progress dots. Example: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`.
- Generic optimize pattern cards under `skills/triton-npu-optimize-knowledge/references/patterns/` are authored Markdown sources, while `skills/triton-npu-optimize-knowledge/references/pattern_index.md` is generated and must be regenerated after editing a pattern card instead of hand-edited. Regenerate with: `uv run python skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py --patterns-dir skills/triton-npu-optimize-knowledge/references/patterns --output skills/triton-npu-optimize-knowledge/references/pattern_index.md`
- CANN extension API pattern cards under `skills/triton-npu-cann-ext-api-patterns/references/patterns/` follow the same card format and are also generated. Regenerate with: `uv run python skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py --patterns-dir skills/triton-npu-cann-ext-api-patterns/references/patterns --output skills/triton-npu-cann-ext-api-patterns/references/patterns/index.md`
- Generic optimize symptom cards under `skills/triton-npu-optimize-knowledge/references/symptoms/` are authored Markdown sources, while `skills/triton-npu-optimize-knowledge/references/symptom_index.md` is generated and must be regenerated after editing a symptom card instead of hand-edited. Regenerate with: `uv run python skills/triton-npu-optimize-knowledge/scripts/build_symptom_index.py --symptoms-dir skills/triton-npu-optimize-knowledge/references/symptoms --output skills/triton-npu-optimize-knowledge/references/symptom_index.md`

## Optimization Patterns

- Every optimize pattern card must begin with a top-level `# <Human Title>` heading before its structured sections.
- Every generic optimize pattern card defined in `skills/triton-npu-optimize-knowledge/references/patterns/` must include `## Summary` and `## Use When`; it may additionally use `## Avoid When`, `## Signals`, `## Related Patterns`, and `## What To Verify After Applying`, with optional `### Code`, `### Profile`, and `### IR` under `## Signals`.
- Pattern-card frontmatter may include `priority: high|normal`; omit it to default to `normal`.
- `priority` is index-rendering metadata, not a replacement for structured sections in the card body.
- The generated `pattern_index.md` must include a `## High Priority Patterns` section that lists cards marked `high`.
- `## Summary` describes **what** the pattern is or does (1–2 sentences). `## Use When` describes **when** to apply it (detection conditions). Keep them orthogonal: the same information must not appear in both sections.
- Every generic optimize symptom card defined in `skills/triton-npu-optimize-knowledge/references/symptoms/` must include `## Summary`, `## Evidence To Confirm`, and `## Candidate Pattern Directions`; it may additionally use `## Common Non-Matches`.
- When existing pattern content already semantically belongs to one of those predefined sections, move it into that section instead of leaving it only in free-form prose.
- Free-form sections are still allowed for examples, background, or architecture notes, but authors should preserve the predefined section names exactly so the pattern index generator can extract them reliably.
- For the concrete optimize pattern card authoring note and regeneration workflow, see `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`.

## Scope Guardrails

- Do not move implementation detail from skills into the CLI unless the CLI truly needs it for orchestration.
- Keep archived IR capture and inspection flows in `skills/triton-npu-analyze-ir/scripts/` instead of growing new CLI subcommands for that workflow.
- Do not couple the project to a single operator style beyond what the existing skills already assume.
- Keep documentation at the overview and workflow level unless a file explicitly needs lower-level implementation detail.
