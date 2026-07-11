# Log Check Pattern Analysis Design

## Summary

Add an informational check (check-10) to the existing `log-check` workflow that analyzes
which optimization patterns were used during each round of operator optimization. The new
check writes its findings to a separate output file (`pattern_analysis.md`) and uses a
two-tier evidence model: explicit (pattern names found in round artifacts) and inferred
(from code diff analysis against pattern signal descriptions).

## Problem

The existing `log-check` command validates optimization round quality (strategy distinctness,
novelty, regression, etc.) but does not report which specific optimization patterns from
the staged knowledge library were actually applied. A user or operator developer has no
direct visibility into whether the optimize agent used `tiling`, `autotune`,
`software-pipeline`, or any of the other 22+ patterns — and whether that choice was
explicitly recorded or must be inferred from code changes.

## Goals

- Add an informational check to the log check prompt that detects pattern usage per round.
- Detect patterns from two evidence sources in priority order:
  1. Round artifacts (`attempts.md`, `summary.md`, `opt-note.md`) — explicit mentions.
  2. Code diff analysis against pattern `## Signals` sections — inference.
- Label each detected pattern as "explicit" or "inferred" with source citation.
- Write results to a separate file (`pattern_analysis.md`) without PASS/FAIL semantics,
  keeping the existing `log_check_result.md` contract intact.
- Reuse existing infrastructure: skill staging, workspace access, code agent capabilities.

## Non-Goals

- Do not add a new CLI command or Python module.
- Do not modify the existing check-1 through check-9 logic or output format.
- Do not add machine-readable pattern matching (this stays prompt-driven).
- Do not require every round to map to a named pattern.

## Decision

### Integration: Prompt-only change

Add check-10 to the `build_log_check_prompt()` function in
`src/helix/log_check/log_check_launcher.py`. The code agent launched by
`log-check` already has:

- The operator workspace as its working directory (`opt-round-N/`, `baseline/`).
- The pattern reference library staged via `+triton-npu-optimize-knowledge`.
- File read and diff capabilities (existing checks 4 and 8 already compare code across rounds).

No new modules, no new CLI commands, no new skill staging rules.

### Separate output file

check-10 writes to `pattern_analysis.md` (workspace-relative), separate from
`log_check_result.md`. This avoids the conflict: `log_check_result.md` enforces
PASS/FAIL on every check section, while check-10 is purely informational.

### Two-tier evidence model

| Tier | Source | Label | When |
|------|--------|-------|------|
| 1 | `attempts.md`, `summary.md`, `opt-note.md` | `explicit` | Pattern name appears in artifact text |
| 2 | Code diff + pattern `## Signals` matching | `inferred` | Tier 1 finds nothing |

Tier 2 compares the operator `.py` file between the current round and its predecessor
(previous round or `baseline/`), then matches the nature of changes against each pattern's
`## Signals` section from `references/patterns/*.md`.

### Prompt additions

Added after check-9 in the prompt string:

- **check-10 section**: Two-tier evidence instructions, output format spec, file path target.
- **Evidence labels**: `explicit` (with source file + round citation) vs `inferred` (with
  diff changes + matched signals).
- **Output structure**: per-round breakdown, inferred rationale, pattern coverage summary,
  novel strategy notes.

### What does not change

- `log_check_result.md` format and PASS/FAIL contract stays the same (checks 1-9).
- `build_log_check_request()` signature unchanged.
- Skill staging (`+triton-npu-optimize-knowledge`, `+triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`) unchanged.
- `run_log_check_batch()` — no changes needed (`pattern_analysis.md` is per-workspace,
  produced alongside `log_check_result.md`).

## Code Changes

One file modified:

### `src/helix/log_check/log_check_launcher.py`

- `build_log_check_prompt()`: Insert check-10 section between check-9 and the output format
  requirements block.

## Rationale

- **Minimal surface area**: One prompt string change, zero new files, zero new dependencies.
- **Reuses proven infra**: The code agent already navigates round directories, reads `.py`
  files, and compares code diffs for checks 4 and 8.
- **Clean contract separation**: Separate output file means no format negotiation with the
  PASS/FAIL regime.
- **Evidence transparency**: Explicit vs inferred labeling lets users audit how patterns
  were identified — critical for trust in automated analysis.
