# Hook Guard Readability Refactor Design

## Goal

Refactor the shared Python guard policy module, the backend-specific `PreToolUse` wrappers, the Codex trace hook, and the OpenCode JavaScript hook so the code is understandable without reverse-engineering the control flow.

This is a readability and maintainability refactor only. It must not change hook behavior, staged file paths, or denial semantics.

## User-Visible Semantics

- Hook behavior stays the same.
- Read-denial behavior stays the same.
- Built-in edit tool phase enforcement stays the same.
- Trace output schema stays the same.

## Problems To Fix

- `evaluate_*` names are too vague for code that is actually deciding whether to deny a tool invocation.
- `candidate` is used for multiple different concepts: raw path text, extracted path references, and resolved filesystem paths.
- The main dispatch logic, read-path extraction, built-in edit enforcement, and trace logic are interleaved in ways that make the scripts hard to follow.
- Comments explain some details, but they do not establish a clear top-down mental model.

## Design

### Naming

- Rename the main shared Python policy entrypoint from `evaluate_payload(...)` to `deny_reason_for_tool_use(...)`.
- Remove the obsolete `evaluate_payload(...)` compatibility wrapper instead of keeping two names for the same decision point.
- Rename helper functions toward intent-revealing verbs such as:
  - `deny_reason_for_path_access`
  - `deny_reason_for_built_in_edit_path`
  - `collect_command_path_references`
  - `collect_shell_wrapper_commands`
- Replace generic `candidate` names with more specific names such as `path_text`, `path_reference`, or `resolved_path` depending on the stage of processing.

### Structure

- Keep both hook implementations in a single file each for now, but reorder them into clear sections:
  - top-level hook entrypoints
  - policy/context building
  - tool dispatch
  - built-in edit enforcement
  - read-path extraction and path checks
  - trace helpers
  - generic low-level utilities
- In the shared Python guard policy module, introduce a small context object so path-check functions do not need to carry long argument lists repeatedly.
- In OpenCode, mirror the same structure with small helper objects and similarly named functions so the Python and JavaScript versions are easier to compare.
- Remove dead helper plumbing that no longer changes behavior, such as path-reference wrappers that always carried default values.
- Keep backend-specific CLI entrypoint behavior in thin Codex and Claude wrappers instead of sharing a single executable hook script.

### Comments

- Add a small number of high-value comments that explain:
  - the two enforcement layers
  - the dispatch order
  - why shell commands are scanned as path references rather than interpreted fully
- Avoid line-by-line commentary and focus on section-level understanding.

## Scope

- No multi-file split in this pass.
- No new hook behavior in this pass.
- No trace schema migration in this pass.
