# Unified Bench Profile Output Root Design

## Summary

Local `run-bench` currently has two different profiler-artifact retention behaviors:

- `msprof` mode can preserve per-case profiler directories through a mode-specific environment variable.
- `standalone` mode always uses auto-cleaned temporary directories and deletes them after parsing.

This change unifies the local benchmark contract so both benchmark modes can preserve profiler output under one shared environment variable.

## Goals

- Introduce a shared local benchmark profiler output variable named `HELIX_BENCH_OUTPUT_DIR`.
- Make local `run-bench --bench-mode standalone` preserve profiler output when that variable is set.
- Keep local `run-bench --bench-mode msprof` artifact retention behavior, but move it to the shared variable name.

## Non-Goals

- Do not change remote benchmark artifact handling.
- Do not change `profile-bench` behavior or directory naming.
- Do not change perf artifact file names or comparison semantics.

## Decision

### Shared environment variable

- Local benchmark runners should first read `HELIX_BENCH_OUTPUT_DIR`.
- If the selected value points to an existing non-directory path, fail explicitly.

### Local `msprof` benchmark retention

- Keep the existing preserved-run layout:
  - one run directory under the configured root
  - one `case-<N>/` directory per benchmark case
- Keep owner-only permissions on created directories so local profiler tools do not inherit permissive `umask` settings.

### Local `standalone` benchmark retention

- When the shared variable is unset, keep the existing auto-cleaned temporary-directory behavior.
- When the shared variable is set:
  - create one preserved run directory under the configured root
  - create one case directory per standalone case under that run directory
  - use sanitized case ids in the case directory names
- Preserve those directories after success or failure so raw profiler artifacts remain inspectable.
- Apply the same owner-only permissions to created directories as the local `msprof` path.

## Verification

- Add local standalone benchmark coverage that verifies the shared variable preserves per-case profiler directories under the configured root.
- Update local `msprof` benchmark coverage so the documented variable name is `HELIX_BENCH_OUTPUT_DIR`.
- Run targeted unit tests plus strict file-scoped `pyright` on modified skill scripts.
