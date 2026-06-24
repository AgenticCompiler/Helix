# Optimize Skill Doc Cleanup Design

## Summary

Tighten `skills/triton/triton-npu-optimize/SKILL.md` without changing optimize behavior.

This cleanup addresses three documentation problems:

1. the skill should declare the layered-analysis model before describing individual layers
2. `compare-perf` guidance is unnecessarily repeated
3. `learned_lessons.md` guidance is spread across multiple sections with overlapping wording

## Goals

- Make the layered-analysis ladder explicit before the per-layer detail.
- Remove duplicate `compare-perf` instructions while preserving the contract that it is the only authority for claimed deltas and speedups.
- Consolidate `learned_lessons.md` guidance into one clear contract block.
- Keep the optimize workflow behavior unchanged.

## Non-Goals

- Do not change optimize runtime behavior.
- Do not change prompt behavior.
- Do not rename optimize artifacts.
- Do not change the existing four analysis levels.

## Proposed Changes

### 1. Add A Short Layered-Analysis Contract Before The Layers

In `skills/triton/triton-npu-optimize/SKILL.md`, add a short summary block ahead of the detailed layer subsections.

It should state that:

- optimize analysis is layered
- the default escalation order is `pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation`
- each round should start at the shallowest level that can justify the next move
- rounds should escalate only when the current level is insufficient
- each round must record the chosen level and why it stayed or escalated

This turns the four layer sections into elaborations of a declared model instead of the first place the model appears.

### 2. Deduplicate `compare-perf`

Keep only two core rules in the validate-and-record section:

- once baseline and round perf artifacts both exist, use the `triton-npu-run-eval` skill to run `compare-perf`
- treat `compare-perf` as the only authority for claimed benchmark deltas and speedups

Keep the supporting sentence that forbids hand-calculated speedups, but remove repeated restatements of the same `compare-perf` instruction.

### 3. Consolidate `learned_lessons.md`

Keep `learned_lessons.md` guidance together as one contract block:

- what it is for
- admission criteria
- examples of good entries
- what belongs in round-local artifacts instead

Reduce repetition in:

- `Outputs`
- `Round Records`
- `Hard Rules`

`Hard Rules` should keep only the one lesson-specific rule that adds something new beyond the contract block, or drop the repeated lines entirely if they add no new constraint.

## Test Impact

Update the doc-contract test that reads `skills/triton/triton-npu-optimize/SKILL.md` so it checks:

- the layered-analysis section declares the escalation ladder before the detailed layers
- the duplicated `compare-perf` line is removed
- `learned_lessons.md` guidance still exists but is no longer redundantly repeated

No runtime or prompt tests should need to change because this is a documentation-only cleanup.
