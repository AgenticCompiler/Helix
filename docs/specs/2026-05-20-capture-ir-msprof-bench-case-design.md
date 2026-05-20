# Capture IR Msprof Bench Case Design

## Summary

- Extend `skills/triton-npu-analyze-ir/scripts/capture_ir.py` so `msprof` benchmark harnesses can accept an explicit `--bench <N>` case selector.
- Keep `standalone` capture unchanged and continue routing it through `standalone_bench_runtime.py run-one`.
- Preserve `allow_abbrev=False` so `--bench` is only accepted as a real supported option, not as a risky abbreviation of `--bench-file`.

## Problem

- The repository's `msprof` benchmark contract uses `--bench <N>` to select one declared benchmark case.
- `capture_ir.py` currently builds the `msprof` execution command as `python <bench> --operator-file <operator>` with no case selector.
- Users who try to pass `--bench <N>` today either hit the old argparse abbreviation bug or, after that fix, receive an unrecognized-argument error even though the `msprof` harness contract expects case selection.

## Goals

- Accept `--bench <N>` in `capture_ir.py`.
- Forward that case index only to `msprof` benchmark execution.
- Keep standalone capture deterministic when no explicit case-selection contract exists.

## Non-Goals

- Do not add `--case-id` support to `capture_ir.py`.
- Do not change replay, archive layout, or remote workspace cleanup behavior.
- Do not change `run-bench` or `profile-bench`; this work is local to IR capture.

## Design

- Add optional `--bench` to the `capture_ir.py` parser with integer type.
- Thread the parsed bench case through local and remote capture entrypoints.
- Update `build_execution_command()` to append `--bench <N>` only when:
  - the resolved benchmark mode is `msprof`, and
  - an explicit bench case was provided.
- Reject `--bench` for `standalone` benches with a short actionable `ValueError`.
- Add focused tests for:
  - parser acceptance of explicit `--bench`
  - local `msprof` command rendering
  - remote `msprof` command rendering

## Verification

- Run `uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v`.
- Run `bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py`.
