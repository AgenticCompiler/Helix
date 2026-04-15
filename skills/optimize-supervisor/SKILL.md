---
name: optimize-supervisor
description: Audit a completed optimize round, repair metadata only when existing evidence already supports it, emit a gate decision, and prepare the next-round handoff without doing open-ended optimization work.
---

# Optimize Supervisor

Audit one completed optimize round and decide whether the optimize session may continue.

Use this skill when the CLI launches a supervisor pass after a worker round has finished.

## Inputs

- The optimize workspace
- The latest completed `opt-round-N/` directory
- `opt-note.md`
- Existing benchmark, profiler, and IR artifacts for the completed round when they already exist

## Outputs

- A gate decision for the completed round
- A short supervisor report
- A next-round brief when continuation is allowed
- Metadata repairs only when those repairs are derived from existing facts

## Required References

- Read the sibling `optimize` skill first for the workflow contract that the worker was supposed to follow.
- Read the latest `opt-round-N/attempts.md`, `opt-round-N/summary.md`, and `opt-round-N/round-state.json` before deciding anything.
- Read `opt-note.md` before changing session-level metadata.
- Read existing round-local benchmark, profiler, and IR artifacts only when they already exist and are needed to verify the worker's recorded claims.
- Do not invoke analysis skills from the supervisor pass as part of the normal workflow.

## Workflow

1. Confirm the latest round has the required artifacts and that the artifacts are internally consistent.
2. Check whether the round records a hypothesis, supporting evidence, correctness status, benchmark status, and comparable perf data.
3. Repair only metadata that can be regenerated from existing facts, such as normalizing `summary.md`, refreshing `opt-note.md`, or producing a clearer handoff brief.
4. Emit one gate decision:
   - `pass-continue`
   - `pass-stop`
   - `revise-metadata`
   - `revise-required`
   - `hard-fail`
5. If continuation is allowed, write a short next-round brief that names the parent round, the suggested next hypothesis, and any evidence that must be collected before more code changes.

## Quality Rules

- Do not perform open-ended optimization work.
- Do not edit the round-local operator implementation unless the calling prompt explicitly asks for a narrow metadata-only repair in the same file and that repair is safe.
- Do not fabricate missing benchmark, profiler, IR, or correctness evidence.
- Do not launch new profiler or IR collection from the supervisor pass.
- Do not mark a round as passing when correctness or benchmark evidence is missing.
- Do not silently promote an invalid round to current best.
- Prefer blocking the session over guessing.
