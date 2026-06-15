# Bench Mode Torch Npu Profiler Rename Design

## Summary

Rename the benchmark mode value `standalone` to `torch-npu-profiler` because the mode no longer means a standalone benchmark-file structure. It now means the runner profiles the shared import-only benchmark contract through `torch_npu.profiler`.

## Goals

- Make benchmark mode naming match current runtime semantics.
- Keep `bench-mode` metadata and CLI mode selection aligned on the same enum values.
- Update runtime code, CLI defaults, prompts, skills, and tests together so no mixed old/new mode names remain.

## Non-Goals

- Do not rename unrelated `test-mode standalone` behavior.
- Do not change `msprof` semantics.
- Do not redesign benchmark execution flow in this rename.

## Decision

- Replace benchmark-mode value `standalone` with `torch-npu-profiler`.
- Keep `bench-mode` as the metadata key and `--bench-mode` as the CLI flag.
- Treat `torch-npu-profiler` and `msprof` as the two supported benchmark profiling strategies.
- Update helper defaults that previously inferred or stored `standalone` benchmark mode so they now use `torch-npu-profiler`.

## Verification

- Add failing parser and metadata contract tests first.
- Run focused CLI, execution, benchmark, profile, and generation-contract tests after the rename.
