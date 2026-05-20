# Capture IR No Argparse Abbrev Design

## Summary

- Make `skills/triton-npu-analyze-ir/scripts/capture_ir.py` reject abbreviated long options such as `--bench`.
- Preserve the existing `capture_ir.py` contract: the script still accepts only `--ir-dir`, `--bench-file`, `--operator-file`, and remote-execution flags.
- Convert the current misleading path-resolution failure into an explicit CLI parse failure before any filesystem work starts.

## Problem

- `capture_ir.py` uses Python `argparse` with the default `allow_abbrev=True`.
- The script defines `--bench-file` but does not define `--bench`.
- When a user appends `--bench 5`, `argparse` treats `--bench` as an abbreviation for `--bench-file` and overwrites the earlier bench-file value with `5`.
- The script then fails later with `FileNotFoundError` for a fake bench-file path, which hides the real mistake.

## Goals

- Reject unsupported abbreviated long options in `capture_ir.py`.
- Keep the IR-capture workflow deterministic and unchanged for valid invocations.
- Lock the behavior with a focused regression test.

## Non-Goals

- Do not add benchmark case-selection support to `capture_ir.py`.
- Do not change local or remote capture semantics.
- Do not broaden the accepted CLI surface.

## Design

- Construct the parser with `allow_abbrev=False`.
- Add a unit test that proves `--bench` now fails during argument parsing instead of being accepted as `--bench-file`.
- Leave the rest of the script unchanged so valid calls behave exactly as before.

## Verification

- Run `uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v`.
- Run `bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py`.
