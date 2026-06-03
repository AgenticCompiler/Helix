# Optimize Round Timing Warning Design

## Summary

- Add a `check-round` warning when the current `opt-round-N` directory appears too close in time to the immediately previous `opt-round-(N-1)` directory.
- Use directory `mtime` as the first implementation's time source.
- Treat this as a warning-style issue on a passing round, not a blocking validation failure.
- Use a fixed threshold of `5` seconds for the initial rollout.

## Problem

During optimize sessions, agents sometimes create multiple rounds in rapid succession by scripting parameter sweeps or generating several round directories as part of one automated loop. That behavior undermines the intended round-by-round workflow, where each round should be evidence-driven and reviewed before the next round begins.

The existing `check-round` contract already returns non-blocking warnings for suspicious but not definitively invalid situations. The "rounds created too quickly" signal fits that category: it is useful evidence of likely bad workflow behavior, but it is still heuristic and should not automatically invalidate an otherwise complete round.

## Design

### Detection Scope

When `check-round` evaluates `opt-round-N`, it should look for the immediately previous round directory `opt-round-(N-1)`.

- If the current round name does not follow the `opt-round-<integer>` pattern, skip the timing check.
- If there is no previous round directory, skip the timing check.
- If either directory timestamp cannot be read, skip the timing check.

### Time Source

Use each round directory's filesystem modification time (`mtime`).

This keeps the first version simple and avoids introducing new round-state fields or parsing timestamps from artifact contents.

### Threshold

If the elapsed time from the previous round directory `mtime` to the current round directory `mtime` is less than `5` seconds, append a warning-style issue to the passing `check-round` result.

The warning should explain that adjacent rounds were created unusually quickly and that this often indicates a scripted parameter sweep or automatic multi-round generation. It should steer the operator back to the intended workflow: review evidence manually and avoid advancing rounds as a batch.

### Result Semantics

- Keep `decision="pass"` when the round otherwise passes validation.
- Add the timing warning to `issues`, alongside existing local-optimum or target-mismatch warnings.
- Let the existing summary builder surface the warning through `guideline`.

This preserves the current `check-round` contract shape and keeps the warning visible to both human users and orchestration code without turning a heuristic signal into a hard gate.

## Non-Goals

- Do not add a new CLI flag or environment variable for the threshold in this change.
- Do not add or require timestamps in `round-state.json`.
- Do not reject rounds solely because they were created quickly.
