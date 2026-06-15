# Unified Bench Generation Spec Design

## Summary

Keep one normative benchmark-generation spec for `triton-npu-gen-bench` because generated `torch-npu-profiler` and `msprof` benchmark files now share the same import-only contract. Preserve `# bench-mode: <torch-npu-profiler|msprof>` as required metadata so the generated file still records its default execution mode.

## Goals

- Remove duplicated benchmark-generation spec text that no longer reflects different file shapes.
- Make the skill describe one shared benchmark contract instead of "standalone style" versus "msprof style".
- Keep `# bench-mode:` explicitly required in both the unified spec and the generation skill.

## Non-Goals

- Do not remove `bench-mode` from generated benchmark files.
- Do not collapse runtime execution behavior; `standalone` and `msprof` still differ at execution time.
- Do not redesign `run-bench`, `profile-bench`, or the benchmark runtime helper in this documentation cleanup.

## Decision

- Replace the mode-specific benchmark spec references with one unified `bench-spec.md`.
- Update `skills/triton-npu-gen-bench/SKILL.md` so it points to the unified spec and explains that generators must set `# bench-mode:` to the requested default mode.
- Keep mode-specific execution notes inside the unified spec only where they describe runtime semantics, not different generated file structures.
- Update contract tests so they validate the unified spec instead of two duplicated spec files.

## Verification

- Re-read the unified spec and generation skill for stale "two styles" wording.
- Run the focused generation-contract test coverage that checks benchmark spec wording.
