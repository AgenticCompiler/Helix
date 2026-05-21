# Capture IR Cmd List Normalization Design

## Summary

- Extend `skills/triton-npu-analyze-ir/scripts/capture_ir.py` so replayed compile commands preserve newer Triton Ascend pass options whose single argument value contains embedded spaces.
- Keep the existing replay contract unchanged: extract `[DEBUG] cmd_list`, normalize it into structured argv tokens, replace the archived input, and replay the compiler with added IR-dump flags.
- Add a focused regression test that covers `--triton-to-linalg=... named-ops=...` style options so the capture workflow no longer turns pass-option fragments into extra positional arguments.

## Problem

- `capture_ir.py` currently parses `[DEBUG] cmd_list: ...` with `shlex.split()`.
- Some Triton Ascend versions emit pass options like `--triton-to-linalg=global-kernel=false named-ops=True enable-select-analysis=False` without shell quoting.
- After `shlex.split()`, the trailing `named-ops=True` and similar fragments become standalone tokens.
- The current `_normalize_compile_command_tokens()` logic only repairs `--append-bisheng-options=...`, so replay leaves those fragments as independent argv items.
- `triton-adapter-opt` then fails with `Too many positional arguments specified!` during IR replay.

## Goals

- Preserve pass options whose value is emitted as `--flag=first=value extra=value ...`.
- Keep genuine positional arguments such as source files, output files, and library paths untouched.
- Fix the regression with a narrow change and a targeted unit test.

## Non-Goals

- Do not redesign the capture workflow or manifest format.
- Do not attempt to normalize arbitrary malformed shell syntax beyond the patterns emitted by Triton Ascend `cmd_list`.
- Do not change remote replay quoting rules except through the normalized argv result.

## Design

- Generalize compile-command normalization from a hard-coded `--append-bisheng-options=` special case to a small allowlist of options known to carry space-separated sub-options inside one argv value.
- When normalization sees one of those option prefixes, keep consuming following tokens while they look like inline sub-options rather than top-level CLI flags or known positionals.
- Treat tokens of the form `name=value` as continuations of the previous option value.
- For `--append-bisheng-options=...`, also continue consuming bare non-flag tokens so paths such as `/opt/lib/libdevice.10.bc` stay attached.
- Leave tokens beginning with `-` as standalone top-level flags so `-o <path>` and other ordinary options still parse normally.

## Verification

- Run `uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v`.
- Run `bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py`.
