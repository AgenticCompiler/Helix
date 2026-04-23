# Optimize Analysis Ownership Design

## Summary

Clarify optimize analysis ownership without splitting skills.

Keep the existing layered optimize ladder:

- `pattern triage`
- `profiling diagnosis`
- `IR attribution`
- `compiler-source escalation`

Keep `triton-npu-analyze-round-performance` as the owner of round-level `perf-analysis.md`, even when the diagnosis deepens from profile-only evidence into IR-backed attribution.

## Problem

The current optimize contract mixes two different concepts:

- the round's current optimize analysis level
- the evidence sources used inside a round diagnosis

This is most visible around IR work:

- `triton-npu-optimize` treats `IR attribution` as a later optimize analysis level
- `triton-npu-analyze-round-performance` treats IR as a deeper explanatory path after profiler evidence

Both statements are individually reasonable, but together they can mislead code agents into thinking that:

- entering `IR attribution` means the round-performance skill no longer owns the diagnosis
- using any IR evidence automatically changes the round's owner or artifact model

That ambiguity weakens round records and makes it harder to tell whether IR was:

- the primary optimize analysis level for the round, or
- supporting evidence used by the round-performance diagnosis flow

## Goals

- Keep the current skill set. Do not split round analysis into two new skills.
- Preserve the existing optimize ladder, including `IR attribution` as a real optimize level.
- Make `triton-npu-analyze-round-performance` the clear owner of `opt-round-N/perf-analysis.md`.
- Clarify that a round may deepen from profile-only evidence into profile-plus-IR evidence without changing artifact ownership.
- Make `attempts.md` and `summary.md` distinguish the round's primary level from its supporting evidence.

## Non-Goals

- Do not add a new optimize runtime state machine.
- Do not rename `perf-analysis.md`.
- Do not remove `triton-npu-analyze-ir`.
- Do not require IR for every performance diagnosis.
- Do not require `perf-analysis.md` for every round.

## Proposed Contract

### 1. Separate level from evidence

The optimize contract should explicitly distinguish:

- `Primary analysis level`: the optimize layer the round is currently operating in
- `Supporting evidence`: the evidence types actually used to justify the diagnosis

This avoids treating "used IR" and "the round is in IR attribution" as synonyms.

### 2. Keep round-performance as the diagnosis owner

`triton-npu-analyze-round-performance` should explicitly support two common completion paths:

- `profile-only diagnosis`
- `profile-plus-IR diagnosis`

Both paths still produce the same round-level output:

- `opt-round-N/perf-analysis.md`

The skill should say clearly that it remains the owner of that artifact when IR is used as deeper explanatory evidence.

### 3. Reframe `triton-npu-analyze-ir` as the IR evidence companion

`triton-npu-analyze-ir` should remain the skill that captures, navigates, and inspects IR artifacts.

Within optimize rounds, its role is:

- capture or inspect `opt-round-N/ir/`
- surface stage-level or lowering-level signals
- provide IR evidence that the round-performance diagnosis can incorporate

It should not implicitly become the owner of `perf-analysis.md`.

### 4. Clarify what `IR attribution` means in optimize

In `triton-npu-optimize`, `IR attribution` should remain a real later analysis level.

However, the docs should make two points explicit:

- a round may use limited IR evidence while still primarily operating as profiler-first diagnosis
- when the round genuinely reaches `IR attribution`, the diagnosis may still be written through `triton-npu-analyze-round-performance`, with `triton-npu-analyze-ir` supplying the IR evidence

This keeps the optimize ladder and the round-analysis workflow compatible instead of forcing a one-to-one mapping between level and skill.

## Artifact Contract Changes

### `attempts.md`

The attempt log should record at least:

- `Primary analysis level`
- `Supporting evidence`
- why that level was chosen
- why the round stayed there or escalated deeper

Suggested evidence examples:

- `code structure + benchmark behavior`
- `profile`
- `profile + .bin`
- `profile + IR`

### `summary.md`

The round summary should also distinguish:

- final or primary analysis level
- which supporting evidence actually decided the round outcome

This keeps later readers from inferring the round level solely from the presence of `ir/` or `perf-analysis.md`.

## Implementation Impact

Update these areas:

- `skills/triton-npu-optimize/SKILL.md`
- `skills/triton-npu-analyze-round-performance/SKILL.md`
- `skills/triton-npu-optimize/references/artifacts.md`
- optimize contract tests that read those files

Prompt changes should stay small and only reinforce the same ownership language where useful.

## Test Impact

Add or update contract tests so they check that:

- optimize distinguishes the primary level from supporting evidence
- round-performance explicitly owns `opt-round-N/perf-analysis.md`
- round-performance explicitly supports both profile-only and profile-plus-IR diagnosis
- optimize documents that `IR attribution` may still use `triton-npu-analyze-round-performance`
- artifacts require both `Primary analysis level` and `Supporting evidence`
