# Optimize Opt-Note Round-Only Design

## Summary

- Restrict `opt-note.md` to round ledger entries plus one final `## Overall Summary`.
- Stop asking optimize agents to write a session-start diagnosis into `opt-note.md` before round 1.
- Record the initial optimization hypothesis and supporting evidence in `opt-round-1/attempts.md` instead.
- Keep deeper diagnosis artifacts round-local, such as `opt-round-N/perf-analysis.md`.
- Preserve current resume behavior: agents should still read `opt-note.md` to recover round history and the final session summary.

## Problem

The current optimize workflow asks the agent to write a short diagnosis before the first code-changing round, and the workflow context also tells resumed runs to read `opt-note.md` before continuing.

That combination gives the initial diagnosis more authority than it deserves:

- the diagnosis is written before the full optimization evidence exists
- resumed runs treat `opt-note.md` as session truth
- later rounds can become anchored to an early guess instead of revising direction from benchmark, profiler, or IR evidence

This makes `opt-note.md` do two jobs that should stay separate:

- durable round/session bookkeeping
- tentative analytical reasoning that may later be disproved

## Goals

- Make `opt-note.md` a clean, durable session ledger.
- Keep `opt-note.md` limited to completed round records and one final overall summary.
- Move the first-round starting hypothesis out of the top-level session note.
- Keep the initial reasoning visible to future readers without elevating it to session-level fact.
- Preserve the existing `## Overall Summary` block and its current role in status reporting.
- Avoid adding a new top-level artifact just to hold tentative reasoning.

## Non-Goals

- Do not remove `## Overall Summary` from `opt-note.md`.
- Do not change optimize status parsing or best-round summary behavior.
- Do not introduce a new top-level session note such as `session-hypothesis.md`.
- Do not require every round to produce `perf-analysis.md`.
- Do not rewrite historical optimize workspaces that already contain top-level diagnosis prose.

## User-Facing Behavior

### `opt-note.md`

For new optimize runs, `opt-note.md` should contain only:

- one concise section per completed round
- one final `## Overall Summary` section at the end

It should no longer contain a session-start diagnosis block, initial bottleneck narrative, or tentative explanation written before `opt-round-1`.

### `opt-round-1/attempts.md`

Before the first code change of round 1, the agent should record:

- the initial optimization hypothesis
- why that hypothesis may help
- what evidence currently supports starting from that direction
- why profiling or IR capture is being skipped, when those tools are not used yet

This content is round-local reasoning, not session-level fact. Later evidence may confirm it, refine it, or invalidate it.

### Resume Semantics

Resumed optimize runs should still read:

- `opt-note.md`
- existing `opt-round-*` directories
- existing round logs

However, reading `opt-note.md` should recover round history and the current overall session outcome, not a top-level diagnosis that biases all future rounds.

## Design

### Artifact Boundary

`opt-note.md` becomes the top-level ledger for durable session outcomes only.

Its responsibilities remain:

- record each completed round concisely
- identify parent round and optimization theme
- record measured outcome and promotion status
- link to `summary.md` and `attempts.md`
- end with one current `## Overall Summary`

Its responsibilities no longer include:

- recording a diagnosis before round 1 starts
- holding tentative bottleneck theories that may later be overturned
- acting as the canonical place for evolving analytical reasoning

### First-Round Hypothesis Placement

The initial diagnosis requirement should be replaced with a round-local hypothesis requirement.

Concretely:

- remove wording that says to write a short diagnosis into `opt-note.md` before the first code-changing round
- keep or strengthen the existing round-lifecycle requirement to create `opt-round-N/attempts.md` immediately
- make round 1 follow the same evidence-first logging pattern as every later round

This keeps the first optimization idea attached to the round that actually tested it.

### Round-Local Analysis

Tentative or evolving reasoning belongs in round-local artifacts:

- `opt-round-N/attempts.md` for the running hypothesis and decision trail
- `opt-round-N/summary.md` for the round conclusion
- `opt-round-N/perf-analysis.md` when a deeper diagnosis is needed

This keeps analysis close to the evidence that produced it and makes later pivots easier to understand.

### Overall Summary Preservation

`## Overall Summary` remains in `opt-note.md` and continues to describe the session outcome, including:

- final best round
- benchmark summary metrics
- validated branches
- outcome
- next step

This block is a session conclusion, not an initial assumption, so it should remain top-level.

### Historical Compatibility

No migration is required for existing workspaces.

- Old `opt-note.md` files that already contain diagnosis prose remain readable.
- Status parsing continues to rely on round markers and the final summary block, not on the removed diagnosis guidance.
- The change affects future optimize guidance and future generated artifacts.

## Implementation Shape

Update optimize workflow documentation and guidance so they all reflect the same artifact boundary:

- `skills/triton/triton-npu-optimize/SKILL.md`
- `skills/triton/triton-npu-optimize/references/workflow.md`
- `skills/triton/triton-npu-optimize/references/opt-note-format.md`
- `skills/triton/triton-npu-optimize/references/artifacts.md`

The key wording changes should:

- delete the instruction to write a pre-round diagnosis into `opt-note.md`
- explicitly say that `opt-note.md` contains only round entries and `## Overall Summary`
- explicitly place the first-round hypothesis in `opt-round-1/attempts.md`
- keep resume instructions that read `opt-note.md` for session continuity

No CLI flag, parser, runtime control-flow, or status-rendering change is required for this design.

## Testing

- Update documentation- or prompt-focused tests that pin optimize workflow wording.
- Add or adjust assertions where needed so future guidance does not reintroduce session-start diagnosis into `opt-note.md`.
- Keep existing status tests that depend on `## Overall Summary`.
- Confirm no test assumes `opt-note.md` must contain pre-round diagnosis text.

## Expected Outcome

- `opt-note.md` stops anchoring future rounds to an early diagnosis guess.
- The first optimization idea remains documented, but only as round-1 reasoning.
- Later rounds are freer to pivot based on measured evidence.
- Top-level optimize notes stay concise and reliable as a session ledger.
