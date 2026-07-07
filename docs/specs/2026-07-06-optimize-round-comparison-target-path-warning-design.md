# Optimize Round Comparison Target Path Warning Design

## Summary

- Rename the round-state field `comparison_target` to `comparison_target_path` so the
  contract makes it explicit that the value is a path.
- Keep round-state loading backward-compatible with legacy `comparison_target` inputs.
- Improve round and baseline check diagnostics for path-bearing round-state fields so
  operators see which field is wrong, what value was found, and what path is expected.

## Goals

- Make the canonical baseline comparison field self-describing in `round-state.json`.
- Preserve compatibility with existing round directories and older skill output.
- Replace terse path mismatch warnings with actionable messages that tell the operator
  how to repair the state file.
- Review neighboring warning strings in the same check flow and improve the ones that
  share the same usability problem.

## Non-Goals

- Do not rename unrelated fields such as `perf_artifact`, `summary_path`, or
  `baseline_operator`.
- Do not introduce an automatic migration command that rewrites existing round-state
  files on disk.
- Do not redesign local-optimum warnings, kernel continuity warnings, or optimize
  workflow-state validation outside the round/baseline artifact checks.

## Problem

The current field name `comparison_target` hides an important fact: the value is not a
label or enum, but a filesystem path that should resolve to the canonical baseline perf
artifact.

The current warning behavior has two usability problems:

1. Path failures are reported in raw, implementation-shaped strings such as
   `comparison_target=... (expected ...)`, which do not explain whether the declared path
   is missing, resolves outside the canonical baseline artifact, or conflicts with the
   current baseline contract.
2. Related path-bearing warnings in the same round/baseline check flow also omit field
   names or current values, for example `missing baseline/perf.txt`,
   `summary_path must be summary.md`, and `perf_artifact must be opt_kernel_perf.txt`.

These diagnostics are technically correct but not friendly to the round operator who now
has to inspect the checker implementation to understand the repair.

## User-Visible Semantics

### New Canonical Field Name

`round-state.json` should now use:

```json
{
  "comparison_target_path": "../baseline/kernel_perf.txt"
}
```

This field continues to mean: "the path from the round directory to the canonical
baseline perf artifact used by the official `compare-perf` decision for the round."

### Legacy Compatibility

Round-state loading should accept the legacy field name `comparison_target` when
`comparison_target_path` is absent.

If both fields are present:

- identical values are accepted
- different values are rejected with a clear validation error that names both fields

The public runtime `RoundState` model should expose `comparison_target_path`, not the
legacy name.

### Improved Path Diagnostics

Path-bearing diagnostics in round and baseline checks should follow these rules:

- name the field explicitly
- include the declared value when one exists
- explain whether the problem is "missing file", "wrong canonical target", or "cannot
  validate because another contract is invalid"
- include the expected canonical path when one is known

Representative message shapes:

- `comparison_target_path points to a missing file: ../baseline/missing_perf.txt (expected ../baseline/kernel_perf.txt)`
- `comparison_target_path must point to the canonical baseline perf artifact ../baseline/kernel_perf.txt (got ../baseline/other_perf.txt)`
- `cannot validate comparison_target_path because baseline/state.json is invalid: missing required baseline-state fields: perf_artifact`

The exact wording can differ, but the message must preserve those three pieces of
information: field name, observed value, repair target.

## Scope Of Warning Improvements

This change should improve the warning families that share the same path-usability issue
inside `skills/common/ascend-npu-optimize-state/scripts/{baseline,round}/check.py`:

- canonical comparison target path resolution
- declared round perf artifact path resolution
- declared round summary path resolution
- declared round perf-analysis path resolution
- declared baseline perf artifact path resolution
- declared baseline operator snapshot path resolution

Warnings that are already sufficiently specific should stay unchanged for now:

- `correctness_status=...`
- `benchmark_status=...`
- `effective_metric_source=...`
- `missing supporting evidence sources`
- kernel continuity diagnostics
- local optimum warnings

This keeps the change focused on path-bearing diagnostics instead of turning it into a
full wording sweep across the entire optimize-state skill.

## Proposed Implementation

### Contract And Model Updates

- Update `skills/common/ascend-npu-optimize-state/references/round-contract.json` to
  define `comparison_target_path` as the required field.
- Update generated or mirrored artifact references that describe round-state required
  fields.
- Rename `RoundState.comparison_target` to `RoundState.comparison_target_path`.
- Update round-state loading so the required-field check treats the new field as
  required, while still accepting legacy input data through compatibility logic in the
  loader.

### Compatibility Logic

`load_round_state()` should normalize the input before constructing `RoundState`:

- read `comparison_target_path` if present
- otherwise fall back to legacy `comparison_target`
- if both exist and differ, raise `ValueError`
- if neither exists, surface the new required-field name in the error

This keeps downstream code and tests aligned around one canonical property name.

### Shared Path-Issue Formatting

Add a small shared helper for path-bearing diagnostics rather than building these strings
inline in each checker branch.

The helper should cover at least:

- missing declared path
- declared path exists but is not the canonical expected target
- "cannot validate X because Y is invalid"

The output should remain concise, but it should stop looking like an internal tuple dump.

### Round Check Changes

Update `check_round()` so the canonical baseline comparison branch reports:

- missing declared comparison target path with the field name
- wrong-but-existing comparison target path with the field name and expected canonical
  target
- upstream baseline invalidation with the original baseline error text

Update `inspect_round_artifacts()` so declared `summary_path`, `perf_artifact`, and
`perf_analysis_path` failures use the same field-aware style.

### Baseline Check Changes

Update `inspect_baseline_artifacts()` so declared baseline path issues mention the owning
field rather than only the missing filename.

Example direction:

- `perf_artifact points to a missing file: baseline/perf.txt`
- `baseline_operator points to a missing file: baseline/kernel.py`

## Documentation And Examples

Update all normative examples and references that still show `comparison_target`:

- round contract reference
- Triton optimize artifact reference
- TileLang optimize artifact reference
- relevant design and plan examples that are meant to be copied by developers
- test fixtures that intentionally model current canonical round-state payloads

Legacy examples may remain only when they explicitly document compatibility behavior.

## Testing

Add or update tests for:

- loading round state with `comparison_target_path`
- loading legacy round state with only `comparison_target`
- rejecting conflicting `comparison_target_path` and `comparison_target`
- round check output for a missing comparison target path
- round check output for a wrong-but-existing comparison target path
- round check output when baseline validation fails before canonical comparison
- artifact inspection output for the neighboring path-bearing warning families that are
  improved by this change

## Rollout

This is a compatibility-preserving contract rename:

- new writers should emit `comparison_target_path`
- existing round directories continue to load
- operators get clearer repair guidance before any future removal of the legacy alias
