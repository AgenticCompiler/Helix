# Helix Rename Design

## Summary

Rename the repository brand from `triton-agent` / `triton_agent` to `helix` across the main CLI, Python package, runtime paths, hooks, packaging assets, the upload-server subproject, tests, and repository documentation.

## Goals

- Rename the installable package and CLI entrypoint to `helix`.
- Rename the main Python package from `triton_agent` to `helix`.
- Rename project-owned runtime names such as environment-variable prefixes, hidden work directories, log directories, staged hook directories, and generated artifact names.
- Rename the upload-server subproject so it stays aligned with the top-level project name.
- Update tests and current documentation so the renamed project is the only user-visible name left in tracked repository content.

## Non-Goals

- Do not rename Triton language references, skill directories, or other domain names where `triton` refers to the compiler/runtime ecosystem rather than this project.
- Do not change workflow behavior beyond what is required to keep the renamed project working.
- Do not refactor unrelated modules or reshape package boundaries.

## User-Visible Behavior

- `uv run helix ...` replaces `uv run triton-agent ...`.
- The upload server runs as `uv run helix-upload-server ...`.
- Project-owned runtime state moves from names like `.triton-agent/` and `triton-agent-logs/` to `.helix/` and `helix-logs/`.
- Project-owned environment variables move from the `TRITON_AGENT_` prefix to `HELIX_`.

## Implementation Notes

- Perform the rename in two layers:
  - mechanical replacement for package names, entrypoints, file paths, and text references
  - targeted follow-up edits where class names, guard strings, or packaging helpers need readable `Helix*` names instead of raw search-and-replace output
- Rename tracked files and directories that embed the old project name, including the main package, PyInstaller spec, hook assets, and upload-server package.
- Verify the repository with search-based checks plus the standard lint, type-check, and test commands after the rename.
