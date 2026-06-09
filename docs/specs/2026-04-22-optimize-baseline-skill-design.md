# Optimize Baseline Skill Design

## Summary

- Introduce a new sibling skill: `triton-npu-prepare-optimize-baseline`.
- Move baseline preparation out of `triton-npu-optimize` and into the new skill.
- Make the new baseline skill responsible for:
  - reusing or generating missing test and benchmark harnesses
  - doing the minimum repair required to reach a correct, benchmarkable starting point
  - creating `baseline/`
  - writing `baseline/state.json` and `baseline/perf.txt`
  - running `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` `check-baseline` until the baseline is reusable
- Rewrite `triton-npu-optimize` so it delegates baseline setup to the new skill when baseline artifacts are missing or invalid.
- Stop teaching `triton-npu-optimize` to call helper scripts directly; the optimize skill should reference owning skills instead of script paths.

## Problem

The current optimize contract still mixes two different phases:

1. baseline preparation
2. optimization rounds

That makes `triton-npu-optimize` broader than it needs to be.

Baseline preparation already includes a full unit of work:

- determine whether existing harnesses can be reused
- generate missing harnesses
- do minimal repair when the starting point is not yet correct or benchmarkable
- establish canonical baseline artifacts
- verify the baseline with `check-baseline`

This is not just a small pre-step. It is a distinct workflow phase with its own completion condition.

The current skill contract also leaks implementation detail upward. In `triton-npu-optimize/SKILL.md`, the agent is still told to use direct script paths such as:

- `../triton-npu-run-eval/scripts/run-command.py`

That violates the repository's preferred boundary. Top-level workflow skills should tell the agent which sibling skill owns a task, not which helper script happens to implement it.

## Goals

- Give baseline preparation a dedicated skill boundary.
- Keep `triton-npu-optimize` focused on round entry, layered analysis, validation, and recording.
- Make the optimize skill call out owning skills rather than helper scripts.
- Preserve the current baseline artifact contract and `check-baseline` gate.
- Preserve the current layered-analysis optimize workflow.

## Non-Goals

- Do not change optimize runtime orchestration in this design.
- Do not remove `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`.
- Do not move artifact contracts out of `references/artifacts.md`.
- Do not weaken the requirement that optimize work must be correctness- and benchmark-validated.
- Do not rewrite the run-eval, generation, profiling, or optimize-check helper scripts.

## Proposed Skill Boundary

### New Skill: `triton-npu-prepare-optimize-baseline`

This new skill should own the baseline-preparation phase end to end.

Its responsibilities should be:

- inspect the operator workspace
- determine whether reusable correctness and benchmark harnesses already exist
- call `triton-npu-gen-test` only when a usable correctness harness is missing
- call `triton-npu-gen-bench` only when a usable benchmark harness is missing
- use `triton-npu-run-eval` for correctness and benchmark validation
- do the minimum repair needed to reach a correct, benchmarkable starting point
- establish canonical baseline artifacts under `baseline/`
- write `baseline/state.json`
- write `baseline/perf.txt`
- call `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` to run `check-baseline`
- keep repairing baseline state until `check-baseline` passes

The new skill should stop after the workspace has a reusable canonical baseline.

### Existing Skill: `triton-npu-optimize`

After this change, `triton-npu-optimize` should no longer own baseline setup details.

It should say:

- if `baseline/` is missing or invalid, first use `triton-npu-prepare-optimize-baseline`
- once baseline exists, enter the optimize round loop
- do round entry
- do layered analysis
- validate and record the round

This makes optimize rounds the primary concern of the optimize skill instead of making it a mixed baseline-plus-rounds contract.

### Existing Skill: `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`

`triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` should stay narrow.

It should continue to:

- verify baseline artifacts with `check-baseline`
- verify round artifacts with `check-round`

It should not become the place that performs open-ended baseline repair.

## User-Facing Behavior

### When Starting Optimize

The optimize workflow should now behave conceptually like this:

1. check whether the workspace already has a reusable canonical baseline
2. if not, use `triton-npu-prepare-optimize-baseline`
3. once baseline is reusable, continue with optimize rounds through `triton-npu-optimize`

This changes the documentation contract, not the conceptual lifecycle.

### Baseline Skill Completion Condition

The new baseline skill should consider its work complete only when:

- the workspace has reusable correctness and benchmark harnesses
- `baseline/` exists
- `baseline/state.json` exists and matches the artifact contract
- `baseline/perf.txt` exists
- `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round check-baseline` passes

## Skill-First Reference Boundary

Top-level workflow skills should prefer naming the owning sibling skill instead of helper scripts.

That means `triton-npu-optimize/SKILL.md` should stop saying things like:

- generate missing tests or benchmarks through `../triton-npu-run-eval/scripts/run-command.py`
- use the bundled helper script for generation, validation, profiling, and comparison commands

Instead, it should say:

- use `triton-npu-prepare-optimize-baseline` to establish baseline when needed
- use `triton-npu-run-eval` for correctness validation, benchmark validation, and `compare-perf`
- use `triton-npu-profile-operator` for profiling diagnosis
- use `triton-npu-analyze-round-performance` for deeper round-local performance analysis
- use `triton-npu-analyze-ir` for IR attribution
- use `triton-npu-analyze-compiler-source` only for the final compiler-source escalation
- use `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` for baseline and round gating

Helper scripts may still be documented inside the owning skill that directly wraps them.

## Documentation Structure Changes

### `triton-npu-optimize/SKILL.md`

Rewrite the baseline section so it becomes a handoff instead of a full procedure.

It should say:

- reuse existing baseline when valid
- otherwise call the baseline-preparation skill
- once baseline exists, continue with round entry and layered analysis

It should no longer directly teach script invocation for generation or validation.

### `triton-npu-prepare-optimize-baseline/SKILL.md`

Create a new skill document that explains:

- workspace inspection
- harness reuse versus generation
- minimum repair for a benchmarkable starting point
- baseline artifact creation
- `check-baseline` gate

This new skill becomes the place where baseline-specific procedural detail belongs.

### `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/SKILL.md`

No major behavior change is needed, but its contract should remain clearly gate-only so it does not overlap with the new baseline-preparation skill.

## Contract And Test Impact

The doc-contract tests should be updated to reflect the new skill boundary.

They should check that:

- the new baseline skill exists
- `triton-npu-optimize/SKILL.md` tells the agent to use the baseline-preparation skill when baseline is missing or invalid
- `triton-npu-optimize/SKILL.md` no longer exposes direct run-eval script paths as top-level workflow instructions
- the optimize skill still names the owning skills for compare-perf, profiling, IR analysis, compiler-source analysis, and optimize-check

## Naming Decision

Use `triton-npu-prepare-optimize-baseline` as the new skill name.

Rationale:

- it makes the scope explicit
- it clearly describes a preparation phase rather than an optimize round
- it avoids ambiguity with generic baseline concepts outside optimize

## Migration Notes

This design is primarily a skill-boundary and documentation-contract change.

Implementation should therefore focus on:

- creating the new baseline skill
- slimming `triton-npu-optimize/SKILL.md`
- updating contract tests
- keeping helper-script detail inside the skills that actually own those scripts

## Open Questions Resolved

- baseline preparation should become a separate skill
- the new skill should generate missing harnesses, not assume they already exist
- optimize should reference sibling skills, not direct script paths, at the top-level contract
