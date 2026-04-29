# Optimize Skill Contract Simplification Design

## Summary

- Make `skills/triton-npu-optimize/SKILL.md` the only primary workflow contract for optimize.
- Delete `skills/triton-npu-optimize/references/workflow.md`.
- Replace the current long numbered optimize workflow with a small set of structured stages.
- Move reference-reading instructions into the relevant stage instead of keeping a separate `Required References` section.
- Reduce duplicated guidance across `Required References`, `Pattern References`, `Default Analysis Ladder`, `Workflow`, and `Quality Rules`.
- Keep `references/artifacts.md` and `references/opt-note-format.md` as focused contracts, not alternate workflow narratives.

## Problem

The current optimize skill has grown into several overlapping documents and sections:

- `SKILL.md` explains inputs and outputs
- `SKILL.md` has a large `Required References` section
- `SKILL.md` has a separate `Pattern References` section
- `SKILL.md` has a separate `Default Analysis Ladder` section
- `SKILL.md` has a long numbered `Workflow`
- `SKILL.md` has a `Quality Rules` section that repeats parts of the workflow
- `references/workflow.md` restates the process again

This creates three concrete problems for code agents:

1. The agent must read the same process rules in multiple places.
2. The agent gets a long flat workflow list instead of a clear stage model.
3. Reference-reading guidance is detached from the moment when the reference is actually needed.

Because optimize skills are already long, every extra top-level section and duplicate workflow file increases context load and raises the chance that the agent will fixate on the wrong part of the contract.

## Goals

- Make `SKILL.md` the sole optimize process contract.
- Present optimize as a short, structured stage model instead of a long flat checklist.
- Keep the layered analysis model visible and central.
- Tell the agent what reference to read at the moment that reference becomes relevant.
- Preserve artifact and note contracts in their dedicated reference files.
- Reduce duplicate process wording so future edits have one obvious home.

## Non-Goals

- Do not change optimize runtime behavior in this design.
- Do not merge artifact contracts into `SKILL.md`.
- Do not delete `references/artifacts.md` or `references/opt-note-format.md`.
- Do not weaken the current layered-analysis workflow.
- Do not remove the existing optimize-check, profile, IR, or compiler-source skill integrations.

## Proposed User-Facing Structure

`SKILL.md` should be rewritten around a small set of sections with one clear purpose each.

Recommended structure:

1. `Goal`
2. `Outputs`
3. `Core Loop`
4. `Stage 0: Baseline Setup`
5. `Stage 1: Round Entry`
6. `Stage 2: Layered Analysis`
7. `Stage 3: Validate And Record`
8. `Round Records`
9. `Hard Rules`

The important change is that the workflow becomes stage-based rather than a single numbered list from beginning to end.

## Section Intent

### Goal

Keep this short.

It should explain:

- what optimize is for
- that it improves the operator itself
- that work proceeds through validated rounds anchored to a canonical baseline

### Outputs

Keep the current output contract, but compress it.

It should list only the durable outputs that matter to a reader deciding what optimize produces:

- `baseline/`
- `opt-round-N/`
- `opt-note.md`
- `learned_lessons.md`
- round-local profile, IR, or compiler-analysis artifacts when needed

### Core Loop

This should be the shortest process summary in the file.

It should describe the optimize mental model in a few bullets:

- establish or reuse baseline
- open a round
- choose the current analysis level
- make one coherent optimization attempt
- validate correctness and performance
- record the outcome

This replaces the need for a long up-front process list.

## Stage Design

### Stage 0: Baseline Setup

This stage should describe:

- reuse existing tests and benchmarks when possible
- generate only missing harnesses
- establish `baseline/` before any completed round exists
- use `triton-npu-optimize-check check-baseline`
- use `references/artifacts.md` when writing baseline state
- use `references/opt-note-format.md` when initializing `opt-note.md`

This is the right place to embed the baseline-related reference reads. They should no longer live in a global `Required References` section.

### Stage 1: Round Entry

This stage should describe:

- creating `opt-round-N/`
- choosing a parent candidate
- starting `attempts.md`
- recording the starting hypothesis
- recording the current analysis level

If the round is the first round, this stage should say that the initial hypothesis belongs in `opt-round-1/attempts.md`, not in `opt-note.md`.

### Stage 2: Layered Analysis

This should become the center of the optimize skill.

It should be organized as four named subsections:

1. `pattern triage`
2. `profiling diagnosis`
3. `IR attribution`
4. `compiler-source escalation`

Each subsection should answer three questions:

- when to enter this level
- what reference or sibling skill to read/use here
- what the round must record before going deeper

This is where reference-reading instructions should live.

#### Pattern Triage

This subsection should say:

- inspect current code structure and benchmark behavior
- read `references/pattern_index.md`
- read a detailed pattern reference only if the index suggests a real match
- do not treat this as permission for blind pattern search

This makes the separate `Pattern References` section unnecessary.

#### Profiling Diagnosis

This subsection should say:

- profiling is the default deeper entrypoint when pattern triage is not enough
- use `triton-npu-profile-operator` or `triton-npu-analyze-round-performance` when profiler-backed diagnosis is needed
- write `opt-round-N/perf-analysis.md` when the deeper round-analysis flow is used

#### IR Attribution

This subsection should say:

- use IR only after profiler-backed symptoms still need explanation
- use `triton-npu-analyze-ir`
- keep IR evidence under `opt-round-N/ir/`

#### Compiler-Source Escalation

This subsection should say:

- use only when compiler source analysis is enabled
- use only after profiler and IR evidence have narrowed a concrete compiler-side question
- use `triton-npu-analyze-compiler-source`
- write `opt-round-N/compiler-analysis.md`

### Stage 3: Validate And Record

This stage should describe:

- correctness before trusting performance
- benchmark execution after correctness passes
- `compare-perf` as the sole source of speedup claims
- `triton-npu-optimize-check check-round`
- updating `summary.md`, `opt-note.md`, and `learned_lessons.md`

This is where validation, comparison, and round-finalization rules should live, instead of being scattered across workflow and quality rules.

## Round Records Section

This section should explain the purpose of each record succinctly:

- `attempts.md`: chronological round log
- `summary.md`: round conclusion and reusable optimization points
- `opt-note.md`: top-level round ledger plus final `## Overall Summary`
- `learned_lessons.md`: strict reusable knowledge only

The detailed field and file contracts stay in `references/artifacts.md` and `references/opt-note-format.md`.

## Hard Rules Section

This section should be short and strict.

It should keep only the rules that are genuinely cross-cutting and easy to misuse, such as:

- optimize the Triton kernel path, not just the wrapper surface
- do not claim success without correctness and benchmark evidence
- do not hand-calculate speedups
- do not begin with blind tiling or launch-parameter search without evidence
- do not put round narrative into `learned_lessons.md`

Rules that simply restate a stage should move into that stage and leave this section.

## Deleting `workflow.md`

Delete `skills/triton-npu-optimize/references/workflow.md`.

Rationale:

- it duplicates the process contract already owned by `SKILL.md`
- code agents are likely to read it whenever it exists
- keeping it forces extra context loading for a redundant workflow narrative
- future workflow changes would otherwise need dual maintenance

After this change:

- `SKILL.md` is the only optimize process contract
- `references/artifacts.md` remains the artifact contract
- `references/opt-note-format.md` remains the note-format contract

## Contract And Test Impact

The generation-contract tests should be rewritten so they assert the new structure directly.

They should check that:

- `SKILL.md` no longer tells the agent to read `references/workflow.md`
- `SKILL.md` contains a structured stage model
- the layered analysis subsections remain present
- reference reads are embedded under the relevant stage rather than under a global `Required References` section
- `workflow.md` no longer exists

Existing tests that read `skills/triton-npu-optimize/references/workflow.md` should be updated or removed.

## Migration Notes

This refactor is documentation-first, but it changes the optimize contract enough that prompts and tests should stay aligned with the new single-source structure.

The implementation should therefore:

- rewrite `SKILL.md`
- delete `references/workflow.md`
- update doc-contract tests
- update any prompt or guidance assertions that still mention reading `workflow.md`

## Open Questions Resolved

- `workflow.md` should be deleted, not stubbed.
- `SKILL.md` should become the sole optimize workflow contract.
- reference reading should be embedded into the stage where the reference is actually needed.
- the optimize workflow should be stage-based and layered, not expressed as one long numbered list.
