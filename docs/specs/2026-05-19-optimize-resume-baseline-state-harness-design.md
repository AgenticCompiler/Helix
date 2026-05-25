# Optimize Resume Baseline-State Harness Design

## Summary

- Make optimize resume detection prefer harness paths declared in `baseline/state.json` when that state is present and valid.
- Keep the existing stem-based harness discovery as a compatibility fallback for older workspaces.
- Require the declared `source_operator` in `baseline/state.json` to match the current resolved optimize input before reusing declared harness paths.

## Motivation

Today `optimize --resume auto|continue` only discovers correctness harnesses by guessing names from the current input stem. That fails for legitimate workspaces where the optimize session was established from one operator filename and later resumed through another compatible filename, even though `baseline/state.json` already records the canonical `test_file`, `bench_file`, and `source_operator`.

## Goals

- Let resumable optimize sessions reuse harnesses declared in `baseline/state.json`.
- Preserve explicit failures for partial or ambiguous optimize state.
- Preserve compatibility with older workspaces that do not have usable baseline state.

## Non-Goals

- Do not change baseline validation semantics.
- Do not change optimize round artifacts or prompt behavior.
- Do not allow resume to silently reuse baseline state for a different operator input.

## Design

- Add a resume-local helper that tries to load `baseline/state.json`.
- If baseline state loads successfully and `source_operator` resolves to the current input path, treat `test_file` as the preferred test harness candidate and `bench_file` as the preferred benchmark harness candidate.
- Only accept declared paths when the files exist and their metadata is readable.
- If baseline state is missing, invalid, points at a different `source_operator`, or declares missing files, fall back to the existing stem-based discovery:
  - `differential_test_<input-stem>.py`
  - `test_<input-stem>.py`
  - `bench_<input-stem>.py`
- Keep explicit failure behavior for:
  - multiple viable test harnesses
  - unreadable metadata
  - missing bench harness after both declared-path and stem fallback checks fail

## Compatibility Rules

- A workspace with valid baseline state and a matching `source_operator` may resume even when the harness names do not match the current input stem.
- A workspace whose `baseline/state.json` names a different source operator must not reuse the declared harnesses.
- Older workspaces without usable baseline state continue to rely on stem naming.

## Verification

- Add focused unit tests for resume classification and resolution.
- Run targeted optimize resume and CLI tests.
- Run repository `ruff`, `pyright`, and unittest verification before completion.
