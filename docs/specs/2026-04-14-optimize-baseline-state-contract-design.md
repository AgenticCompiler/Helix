# Optimize Baseline State Contract Design

## Summary

- Make the optimize baseline contract explicit to launched code agents instead of only enforcing it in CLI validation.
- Reuse one shared baseline-state contract description across optimize prompts and temporary optimize guidance files so those surfaces do not drift.
- Keep `resume continue` validation strict, but include the concrete baseline failure detail in the user-facing error.

## Problem

- The optimize runtime currently validates `baseline/state.json` strictly, including required fields such as `baseline_established`, `correctness_status`, and `benchmark_status`.
- The optimize prompts and staged optimize guidance tell the agent to create `baseline/state.json`, but they do not spell out the required field-level schema.
- That leaves the agent without an explicit contract for what to write, even though later CLI validation depends on those fields.
- When `resume continue` fails baseline validation, the CLI currently collapses the detailed baseline issue into a generic `requires established baseline/` error, which makes the root cause harder to diagnose.

## Goals

- Tell the optimize agent exactly which `baseline/state.json` fields must be written and what they represent.
- Keep prompt-level and staged-guidance wording aligned by reusing one code-owned contract renderer.
- Preserve the existing baseline validation rules.
- Surface the first concrete baseline validation failure in `resume continue` errors.

## Non-Goals

- Do not change the required baseline-state fields.
- Do not relax `resume continue` semantics.
- Do not introduce a new optimize subcommand for baseline preparation.

## Design

### Shared Contract Text

- Add a small helper in the optimize baseline layer that renders user-facing baseline-state contract lines.
- Reuse that helper from:
  - optimize worker prompt construction
  - optimize unsupervised prompt construction
  - staged optimize worker guidance

The contract text should explicitly name every required `baseline/state.json` field:

- `baseline_kind`
- `source_operator`
- `baseline_operator`
- `test_file`
- `test_mode`
- `bench_file`
- `bench_mode`
- `perf_artifact`
- `correctness_status`
- `benchmark_status`
- `baseline_established`

It should also say that baseline preparation is complete only after:

- `correctness_status` is `passed`
- `benchmark_status` is `passed`
- `baseline_established` is `true`

### Skill Documentation

- Update the optimize skill and its workflow reference so the natural-language workflow also names the required `baseline/state.json` fields instead of only naming the file path.
- Keep the skill wording consistent with the runtime contract, but do not duplicate lower-level parser or implementation detail beyond the required field list and completion semantics.

### Resume Error Detail

- Preserve the top-level `resume continue requires established baseline/: <path>` wording.
- Append the concrete failure detail from baseline inspection or state parsing, for example:
  - missing `baseline/state.json`
  - missing required baseline-state field names
  - `baseline/state.json marks baseline as not established`

## Testing

- Add prompt tests that verify optimize worker prompts include the required baseline-state fields and completion semantics.
- Add optimize guidance tests that verify staged worker guidance includes the same contract.
- Update CLI resume validation tests to assert the concrete baseline failure detail is preserved in the surfaced error.

## Expected Outcome

- Agents will be told exactly what to write into `baseline/state.json` before optimize rounds begin.
- Prompt and staged-guidance wording will stay aligned more easily.
- Users will get actionable `resume continue` errors when a baseline exists but is incomplete or malformed.
