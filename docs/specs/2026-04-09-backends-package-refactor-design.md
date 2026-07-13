# Backends Package Refactor Design

> **Note:** This spec describes the initial backends package with five backends (`codex`, `opencode`, `pi`, `claude`). The package has since expanded to also include `openhands` and `traecli`. The package structure described here remains valid.

## Summary

- Replace the flat backend runner modules with a `helix/backends/` package.
- Keep backend behavior, CLI flags, skill staging behavior, and optimize resume semantics unchanged.
- Do not introduce a plugin registry or runtime backend discovery in this refactor.

## Goals

- Make the backend-specific code easier to navigate now that the project supports multiple code agents.
- Group the shared backend abstraction, factory, and concrete runner implementations under one explicit package boundary.
- Mirror the recent `generation/` package refactor with the same conservative scope and minimal semantic change.

## Non-Goals

- Do not change user-visible command behavior.
- Do not move workspace skill staging into the backend package.
- Do not add backend capability metadata, registries, or plugin loading.
- Do not refactor unrelated execution, generation, or optimize logic beyond import updates.

## Proposed Package Shape

- `src/helix/backends/__init__.py`
  - stable export surface for backend runner types and factory helpers
- `src/helix/backends/base.py`
  - `AgentRunner`
- `src/helix/backends/factory.py`
  - `create_runner()`
- `src/helix/backends/codex.py`
  - `CodexRunner`
  - `_UnifiedDiffFilter`
- `src/helix/backends/opencode.py`
  - `OpenCodeRunner`
- `src/helix/backends/pi.py`
  - `PiRunner`
- `src/helix/backends/claude.py`
  - `ClaudeRunner`

## Why This Shape

- The existing backend code already behaves like a subdomain: one abstraction, one factory, and one concrete implementation per backend.
- A dedicated package makes that boundary visible without making the CLI thicker.
- Keeping the refactor to packaging and import cleanup preserves the repository rule that backend-specific launch construction stays isolated from CLI parsing and prompt construction.

## Import Migration

- Update repository imports from:
  - `helix.agent`
  - `helix.runner_factory`
  - `helix.codex_runner`
  - `helix.opencode_runner`
  - `helix.pi_runner`
  - `helix.claude_runner`
- Replace them with imports from `helix.backends` or the focused package modules.
- Delete the old flat modules after the package migration is complete.

## Testing

- Update runner tests to import from the new package modules.
- Add direct factory coverage if needed so the package exports are exercised explicitly.
- Run at least:
  - `uv run python -m unittest tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner tests.test_process_runner tests.test_cli -v`
  - `uv run pyright`
