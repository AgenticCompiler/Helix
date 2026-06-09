# Optimize Single-Repair Policy Design

## Goal

Simplify batched optimize repair handling so each revise path gets at most one repair rerun.

## User-Visible Semantics

- When a checked or supervised batch returns `revise-required` or `revise-metadata`, the CLI should launch exactly one repair follow-up worker batch.
- If that repair follow-up still ends in `revise-required` or `revise-metadata`, the CLI should stop immediately and surface a failure instead of looping again.
- A successful pass resets the repair state, so later independent batches may still use one repair follow-up if needed.

## Implementation Notes

- Replace `repair_attempts` counting with a simpler boolean repair-state flag in the batched optimize controller.
- Keep the existing follow-up summary behavior and supervisor-report handoff behavior unchanged.
- Update the repair-loop regression test to assert that the controller stops after the second worker run rather than after multiple retries.
