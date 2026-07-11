# Run-Bench Output And Analysis Hints Design

## Summary

- Align both `run-bench` entrypoints so users can pass `--output` to choose the perf artifact path explicitly.
- Make successful `capture-ir` runs print a next-step hint that points users at the existing IR inspection helper and reminds them that raw archived files remain available.
- Make successful `profile-bench` runs print a next-step hint that points users at the existing `run-command.py profile-report` entrypoint and reminds them that the copied-back profiler files remain available.

## Problem

- The main CLI `helix run-bench` already accepts `-o/--output`, but the skill-local `skills/triton-npu-run-eval/scripts/run-command.py run-bench` parser does not expose the same option.
- `capture-ir` currently stops after printing `Capture manifest: ...`, which leaves users without a clear next step for navigating the archived IR.
- `profile-bench` currently prints the profile directory and an inline summary, but it does not tell users how to re-render that summary later or where to look when the summary is insufficient.

## Goals

- Keep the main CLI and the skill-local `run-bench` entrypoint aligned on `--output`.
- Preserve the existing perf artifact format and default naming when `--output` is omitted.
- Add lightweight success-only hints for IR inspection and profile re-reporting.
- Tell users that the helper summary is optional and that they can still inspect the underlying files directly.

## Non-Goals

- Do not add new top-level CLI aliases such as a new `inspect ir` or `profile report` command surface.
- Do not change perf artifact contents, profile directory layout, or IR archive layout.
- Do not change failure-path output contracts for `run-bench`, `capture-ir`, or `profile-bench`.

## Decision

### `run-bench --output`

- Keep `helix run-bench -o/--output <path>` as the user-facing CLI contract.
- Add `--output <path>` to `skills/triton-npu-run-eval/scripts/run-command.py run-bench`.
- Thread the parsed output path through both local and remote bench execution paths exactly as the main CLI already does today.
- Continue printing the final resolved artifact path as:
  - `Perf file: <abs-path>`
- When `--output` is omitted, keep the current derived perf path behavior unchanged.

### `capture-ir` hint

- After a successful `capture-ir` run prints `Capture manifest: <path>`, print one concise hint that:
  - recommends the bundled `inspect_ir.py` helper plus the captured `--ir-dir <path>` as the first inspection path
  - states that users can inspect the archived raw files directly when they need more detail
- The hint should reference the archive contents that matter most for direct inspection:
  - `bishengir_stages/`
  - `triton_dump/`
  - `all-ir.txt`
  - `capture-manifest.json`
- Print this hint only after the archive has been created successfully.

### `profile-bench` hint

- After a successful `profile-bench` run prints `Profile directory: <path>` and its inline summary, print one concise hint that:
  - recommends the bundled `profile-report` helper plus the emitted `--profile-dir <path>` for re-rendering or exporting the summary later
  - states that users can inspect the copied-back raw profiler files directly when the summary is not enough
- Keep the existing inline summary behavior unchanged.
- Print this hint only when a profile directory was produced successfully.

## Verification

- Add or update tests that confirm main CLI `run-bench` threads `--output` to the bench runner.
- Add or update tests that confirm skill-local `run-command.py run-bench` exposes `--output` in help and threads it to local and remote bench runners.
- Add or update tests that confirm successful `capture-ir` prints the new inspection hint.
- Add or update tests that confirm successful `profile-bench` prints the new `profile-report` hint.
- Update the minimum relevant docs and references so examples mention `--output` where appropriate and clarify that raw artifact inspection remains available when helper summaries are insufficient.
