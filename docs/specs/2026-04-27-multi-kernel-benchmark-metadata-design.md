# Multi-Kernel Benchmark Metadata Design

## Summary

Some Triton NPU operators launch more than one Triton kernel inside a single public operator entrypoint. The current benchmark metadata contract records only one kernel name, which is not enough for `msprof` benchmark aggregation. This change makes benchmark metadata support multiple kernels while keeping `profile-bench` aligned with the `msprof op --kernel-name=<name>` single-kernel limitation.

## Goal

- Let generated benchmark metadata describe one or more Triton kernels.
- Make `run-bench --bench-mode msprof` aggregate profiler latency across all declared kernels for a benchmark case.
- Keep `profile-bench --bench-mode msprof` single-kernel, with explicit runtime kernel selection when needed.

## Decision

### Metadata contract

- Introduce `# kernels: <kernel-a>, <kernel-b>, ...` as the new benchmark metadata field.
- Treat `# kernels:` as the canonical multi-kernel field for newly generated benchmark harnesses.
- Keep runtime parsing backward-compatible with existing `# kernel: <kernel>` metadata by normalizing it to a one-element kernel list.
- Do not require benchmark metadata to also carry a separate default profiling kernel. Profiling selection stays a runtime concern.

### `run-bench --bench-mode msprof`

- Read the normalized kernel list from benchmark metadata.
- After reading `op_statistic_*.csv`, find rows whose `OP Type` exactly matches each declared kernel name.
- Compute `latency-case-<N>` as the sum of matched `Avg Time(us)` values across all declared kernels.
- If some declared kernels are missing from the CSV, sum only the matched subset and still record the full raw op statistics payload.
- If none of the declared kernels are present, emit `latency-case-<N>: NA` and keep the existing raw-op-statistic fallback path for later comparison.

### `profile-bench --bench-mode msprof`

- Add an optional runtime flag: `--kernel-name <name>`.
- If `--kernel-name` is provided, require that it exactly matches one declared kernel from benchmark metadata before invoking `msprof op --kernel-name=<name>`.
- If `--kernel-name` is omitted and benchmark metadata resolves to exactly one kernel, use that kernel automatically.
- If `--kernel-name` is omitted and benchmark metadata resolves to multiple kernels, fail explicitly with an actionable message telling the agent to rerun `profile-bench` with `--kernel-name`.
- This keeps benchmark profiling compatible with the underlying `msprof op` command shape and avoids encoding a second source of truth in metadata.

## Scope

Update the following areas:

- benchmark metadata generation guidance and benchmark spec references
- benchmark/profile runtime helpers under `skills/triton-npu-run-eval/scripts/`
- CLI command parsing/help for `profile-bench --kernel-name`
- repository docs and tests that currently describe or enforce single-kernel benchmark metadata

Do not add a new CLI subcommand. Do not change standalone benchmark behavior. Do not change the core `compare-perf` artifact format beyond the already supported `latency-case-*` plus comment payload contract.

## Error Handling

- If benchmark metadata contains neither `# kernels:` nor `# kernel:`, fail explicitly in `msprof` benchmark and profiling flows.
- If `# kernels:` parses to an empty list after trimming separators and whitespace, fail explicitly.
- If `profile-bench --kernel-name` is provided but is not present in benchmark metadata, fail explicitly and list the declared kernel names.
- If `profile-bench` omits `--kernel-name` for a multi-kernel benchmark, fail explicitly and instruct the caller to pass `--kernel-name`.

## Verification

- Add metadata parsing tests that cover:
  - new `# kernels:` parsing
  - backward compatibility for old `# kernel:`
  - validation failures for missing or empty kernel metadata
- Add `run-bench` msprof tests that cover:
  - summing multiple matched kernels
  - partial matches
  - no matches producing `NA`
- Add `profile-bench` msprof tests that cover:
  - explicit `--kernel-name` success
  - automatic selection for a single declared kernel
  - multi-kernel failure without `--kernel-name`
  - invalid `--kernel-name` rejection
- Update generation contract tests and docs so new benchmark harnesses require `# kernels:` instead of `# kernel:`.
