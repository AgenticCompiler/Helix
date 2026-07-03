# Optimize Round Strategy State Design

## Summary

- Extend the existing optimize workflow state under `.triton-agent/state.json` with round-level strategy fields:
  - `round_strategy`
  - `analysis_policy`
  - `reason`
- Keep the existing lifecycle phases unchanged:
  - `baseline`
  - `awaiting_round_start`
  - `round_active`
- Upgrade `ascend-npu-optimize-state` from baseline/start/submit gating only into the single workflow-state write surface for optimize rounds.
- Extend `start-round` so it initializes the current round strategy state when the round becomes active.
- Add a new `set-current-round-state` subcommand that updates the active round strategy state without taking `--round-dir`.
- Keep `.triton-agent/state.json` as the authority for the latest current state only; record state-change history by appending structured entries to `opt-round-N/attempts.md`.
- Keep `submit-round` as the round-closing command only; it may emit next-round hints but must not write the next round state.

## Goals

- Make the current optimize round strategy explicit and machine-readable.
- Let optimize workflows distinguish the round's optimization intent from the workflow phase.
- Preserve the existing one-round-at-a-time workflow while making state changes easier for agents and scripts to audit.
- Reduce duplicated state prose across `attempts.md`, `summary.md`, and prompts by making the workflow state authoritative.
- Keep the common optimize-state skill as the single place that mutates temporary optimize workflow state.

## Non-Goals

- Do not redesign the lifecycle phases `baseline`, `awaiting_round_start`, or `round_active`.
- Do not combine this design with the separate fast benchmark or `probe-bench` work.
- Do not redesign `baseline/state.json` or `opt-round-N/round-state.json`.
- Do not make `submit-round` automatically choose and persist the next round state.
- Do not introduce a new JSON archive for round strategy history.
- Do not move optimize workflow logic from skills into `src/triton_agent/`.

## Problem

The current optimize workflow state only tracks temporary lifecycle ownership:

- whether baseline is pending or accepted
- whether a round is active
- which round number is active
- round start and end timestamps

That is enough for round gating, but it is not enough to guide the optimize workflow itself.

Today, optimize round intent and analysis depth are implied through free-form prose in prompts and round notes. This creates four concrete problems:

1. The current round's optimization strategy is not machine-readable.
2. The current round's minimum required analysis depth is not machine-readable.
3. Agents must repeat similar state information in `attempts.md` and `summary.md`.
4. There is no single workflow-owned command for updating the active round's strategy when evidence changes mid-round.

The result is that the workflow knows which round is active, but it does not know what the round is trying to do or what level of evidence the round is supposed to require before major code changes.

## User-Visible Semantics

### Lifecycle Phases Stay The Same

The optimize workflow lifecycle does not change:

- `baseline`
- `awaiting_round_start`
- `round_active`

This design adds a round-scoped strategy layer on top of those phases instead of replacing them.

### New Round Strategy State

When a round becomes active, the workflow state should also record:

- `round_strategy`
- `analysis_policy`
- `reason`

These fields belong to the active round entry under `.triton-agent/state.json`.

### New Optimize-State Commands

The public skill entrypoint remains:

```bash
python3 scripts/cli.py <subcommand> ...
```

The skill surface should become:

```bash
python3 scripts/cli.py submit-baseline --baseline-dir baseline
python3 scripts/cli.py start-round --round-dir opt-round-1 --round-strategy exploration --analysis-policy pattern_entry --reason "..."
python3 scripts/cli.py set-current-round-state --round-strategy structural_change --analysis-policy profile_required --reason "..."
python3 scripts/cli.py submit-round --round-dir opt-round-1
```

### Attempts And Summary Responsibilities

`attempts.md`

- remains the round-local chronological log
- becomes the human-readable history of strategy-state changes
- receives structured state update blocks appended by scripts

`summary.md`

- remains the final round conclusion
- should not duplicate the full state-change history
- may mention the final effective strategy state once, but it is not the history ledger

`.triton-agent/state.json`

- remains the authority for the latest current workflow state
- does not become a historical event log

## State Model

### Workflow State Shape

Keep `schema_version` at `1`.

This change is additive: it extends round entries with a `strategy_state` block rather than replacing the workflow shape.

Recommended round entry shape:

```json
{
  "status": "active",
  "round_dir": "opt-round-3",
  "started_at": "2026-06-27T10:00:00Z",
  "ended_at": null,
  "strategy_state": {
    "round_strategy": "focused_tuning",
    "analysis_policy": "ir_required",
    "reason": "Round 2 showed a clear gain, but profiler evidence was not enough to explain the remaining bottleneck.",
    "updated_at": "2026-06-27T10:12:00Z",
    "updated_by": "set-current-round-state"
  }
}
```

### Legacy Session Handling

Because this design keeps `schema_version` at `1`, existing workflow-state files must remain readable.

Compatibility rule:

- existing round entries without `strategy_state` remain valid legacy entries
- new `start-round` invocations must always write `strategy_state`
- `set-current-round-state` may initialize missing `strategy_state` on a legacy active round

That legacy initialization should count as a real state write rather than a no-op.

This keeps the extension deployable without forcing a dedicated workflow-state schema migration.

### `round_strategy`

`round_strategy` answers: what kind of optimization task this round is currently trying to complete.

Allowed values:

- `exploration`
- `structural_change`
- `focused_tuning`
- `stabilization`
- `plateau_review`

Definitions:

`exploration`

- used when the round still needs to narrow the next promising direction
- emphasizes choosing the next coherent hypothesis

`structural_change`

- used when the bottleneck appears to require a larger structural rewrite
- emphasizes shape, layout, dataflow, or algorithm-path changes

`focused_tuning`

- used when the direction is already validated and the round is refining within that direction
- emphasizes narrow, deeper tuning rather than searching for a new path

`stabilization`

- used when the current direction still looks promising but the implementation is unstable
- emphasizes repairing correctness, compilation, runtime stability, or fragile performance before further tuning

`plateau_review`

- used when the current direction is likely near a local plateau
- emphasizes deciding whether to raise analysis depth, change direction, or stop

### `analysis_policy`

`analysis_policy` answers: how deep the evidence must go before the round should make its main code changes.

Allowed values:

- `pattern_entry`
- `profile_required`
- `ir_required`
- `compiler_source_required`

Definitions:

`pattern_entry`

- start from code structure and pattern triage
- profiler evidence is optional, not required before the main edit

`profile_required`

- profiler evidence is required before the main edit

`ir_required`

- profiler evidence alone is not enough; IR attribution is required before the main edit

`compiler_source_required`

- profiler and IR evidence have already narrowed the issue to a compiler-side question
- compiler source evidence is required before the main edit

### `reason`

`reason` is required for all state-initialization and state-update commands.

It provides:

- a human-readable explanation for the chosen or updated strategy
- the text mirrored into `attempts.md`
- the bridge between machine-readable state and round-local narrative

### `updated_by`

Allowed values:

- `start-round`
- `set-current-round-state`

No other command should claim ownership of strategy-state mutation in v1.

## Command Design

### `start-round`

#### Purpose

- enforce the existing workflow phase gate
- open the requested durable `opt-round-N/`
- initialize its strategy state
- create or append the first structured state block in `attempts.md`

#### Public Command

```bash
python3 scripts/cli.py start-round \
  --round-dir opt-round-3 \
  --round-strategy focused_tuning \
  --analysis-policy ir_required \
  --reason "Round 2 showed a clear gain, but profiler evidence was not enough to explain the remaining bottleneck."
```

#### Required Arguments

- `--round-dir`
- `--round-strategy`
- `--analysis-policy`
- `--reason`

#### Behavior

- keep the current baseline and phase checks
- if the round opens successfully, write `strategy_state`
- if `opt-round-N/attempts.md` does not exist yet, create it
- append a structured `State Update` block to `attempts.md`

#### Output

Successful JSON should include:

- `status`
- `round`
- `guideline`
- `hard_rules`
- `round_strategy`
- `analysis_policy`
- `reason`
- optional `warnings`

### `set-current-round-state`

#### Purpose

- update the active round's strategy state mid-round
- never select a round by path
- always target the current workflow-owned active round

#### Public Command

```bash
python3 scripts/cli.py set-current-round-state \
  --round-strategy stabilization \
  --analysis-policy ir_required \
  --reason "The current direction still looks promising, but correctness and performance are unstable and need repair before further tuning."
```

#### Required Arguments

- `--reason`
- at least one of:
  - `--round-strategy`
  - `--analysis-policy`

#### Behavior

- fail if no round is active
- fail if both fields are unchanged
- fail if only `reason` changes and both strategy fields stay the same
- if the active round is a legacy entry with no `strategy_state`, treat the call as initialization instead of a no-op
- update `.triton-agent/state.json`
- append a structured `State Update` block to the active round's `attempts.md`

#### Output

Successful JSON should include:

- `status`
- `round`
- `guideline`
- `round_strategy` as `from -> to`
- `analysis_policy` as `from -> to`
- `reason`
- optional `warnings`

### `submit-round`

#### Purpose

- validate and submit a completed round
- close the active round
- optionally suggest the next round state

#### Behavior Changes

`submit-round` keeps its current ownership:

- validate durable round artifacts
- mark the round complete in workflow state
- return pass/fix guidance

It should not:

- mutate the next round state
- overwrite the current round strategy state before closing
- append strategy history to a new archive JSON

It may optionally add:

```json
"next_round_hint": {
  "round_strategy": "focused_tuning",
  "analysis_policy": "ir_required",
  "reason": "..."
}
```

This hint is advisory only.

## Transition Rules

### `analysis_policy` Is A Hard Discipline Gate

Within one active round, `analysis_policy` may stay the same or become deeper, but it may not become shallower.

Allowed same-round upgrades:

- `pattern_entry -> profile_required`
- `profile_required -> ir_required`
- `ir_required -> compiler_source_required`

Rejected same-round rollbacks:

- `profile_required -> pattern_entry`
- `ir_required -> profile_required`
- `compiler_source_required -> ir_required`

Rationale:

- `analysis_policy` is the minimum evidence depth for the round
- once the round raises that floor, the floor should not drop again in the same round

### `round_strategy` Uses Soft Guidance

`round_strategy` should be more flexible than `analysis_policy`.

Encouraged transitions:

- `exploration -> structural_change`
- `exploration -> focused_tuning`
- `exploration -> plateau_review`
- `structural_change -> focused_tuning`
- `structural_change -> stabilization`
- `focused_tuning -> stabilization`
- `focused_tuning -> plateau_review`
- `stabilization -> focused_tuning`
- `stabilization -> plateau_review`

Allowed but warning-worthy transitions:

- `structural_change -> exploration`
- `focused_tuning -> exploration`
- `plateau_review -> focused_tuning`
- `plateau_review -> structural_change`

Rejected transitions:

- no-op updates
- unknown enum values
- any update without `reason`

The warning-worthy transitions should still succeed, but the command output should carry a `warnings` list and the state-update block written to `attempts.md` should include those warnings.

### Strategy And Policy Combination Warnings

Some combinations are unusual and should warn rather than fail.

Examples:

- `exploration + compiler_source_required`
- `structural_change + pattern_entry`
- `plateau_review + pattern_entry`

These combinations are not impossible, but they should be surfaced as unusual because they often indicate an over-deep or under-deep analysis choice for the stated round intent.

### `start-round` Initialization Is Not A Same-Round Transition

The transition rules above do not constrain `start-round` initialization.

A new round may start from any legal combination of:

- `round_strategy`
- `analysis_policy`

The command may still warn on unusual combinations, but it should not block initialization solely because a combination is unusual.

## Attempts And Summary Contract

### `attempts.md`

`attempts.md` becomes the only required history log for strategy-state changes.

Every successful `start-round` or `set-current-round-state` command should append a structured block.

Recommended initialization block:

```md
## State Update 2026-06-27T10:00:00Z
- Source: start-round
- Round strategy: focused_tuning
- Analysis policy: ir_required
- Reason: Round 2 showed a clear gain, but profiler evidence was not enough to explain the remaining bottleneck.
```

Recommended update block:

```md
## State Update 2026-06-27T10:12:00Z
- Source: set-current-round-state
- Round strategy: structural_change -> focused_tuning
- Analysis policy: profile_required -> ir_required
- Reason: Profiler narrowed the bottleneck, but IR is still needed before the next code change.
```

For template stability, when only one state field changes, still render the other field in `from -> to` form even if its value is unchanged.

Example:

```md
- Analysis policy: ir_required -> ir_required
```

This keeps the script output format stable and makes state blocks easy to scan and parse.

### `summary.md`

`summary.md` should not duplicate the full strategy-state history.

It should remain focused on:

- final outcome
- decisive evidence
- unresolved risks
- why the round should continue or stop

If needed, it may mention the final effective strategy once, but the state-change history belongs in `attempts.md`.

### No New State Archive JSON

Keep the existing timing archive behavior unchanged.

Do not add a new JSON archive for strategy-state history.

The state history source of truth should be:

- latest current state in `.triton-agent/state.json`
- chronological strategy-state blocks in `opt-round-N/attempts.md`

## Documentation And Prompt Migration

### `ascend-npu-optimize-state` Skill

Update the skill so it documents four subcommands:

- `submit-baseline`
- `start-round`
- `set-current-round-state`
- `submit-round`

Its documentation should explicitly state that this skill is the workflow-owned writer for temporary optimize round strategy state.

### `{language}-npu-optimize` Skills

Update both Triton and TileLang optimize skills so they explain:

- the active round strategy state is initialized through `ascend-npu-optimize-state start-round`
- later same-round state changes use `set-current-round-state`
- structured state blocks in `attempts.md` are script-written
- agents do not need to manually duplicate the same `round_strategy`, `analysis_policy`, and `reason` prose in both `attempts.md` and `summary.md`

The skills should still require agents to record:

- hypotheses
- pattern candidates and pivots
- evidence sources and reused paths
- code changes
- correctness failures
- benchmark outcomes

### Prompt Guidance

Prompt text that currently says "record the current analysis level" should be refined so that:

- workflow state is the authority for the active round's latest strategy state
- `attempts.md` contains the structured history mirror written by scripts
- free-form round prose still explains why the round changed direction, but it does not need to restate the same state fields as manual bookkeeping

## Implementation Boundaries

- Keep all state mutation logic under `skills/common/ascend-npu-optimize-state/scripts/state_manage/`.
- Extend `scripts/cli.py` with one new public subcommand: `set-current-round-state`.
- Extend `state_manage/workflow.py` with helpers for:
  - validating strategy-state enums
  - initializing round strategy state on `start-round`
  - updating the active round strategy state
  - validating same-round transitions
  - appending structured state blocks to `attempts.md`
- Keep `src/triton_agent/optimize/workflow_state.py` as a loader bridge only.
- Do not move round strategy logic into `src/triton_agent/`.

## Acceptance Criteria

- `start-round` requires `--round-strategy`, `--analysis-policy`, and `--reason`.
- `start-round` writes `strategy_state` into the active round entry.
- `start-round` creates or appends `opt-round-N/attempts.md` with a structured initialization block.
- `set-current-round-state` fails when there is no active round.
- `set-current-round-state` fails on no-op updates.
- `set-current-round-state` rejects `analysis_policy` rollback.
- `set-current-round-state` may initialize missing `strategy_state` on a legacy active round.
- `set-current-round-state` updates `.triton-agent/state.json` and appends a structured block to `attempts.md`.
- warning-worthy `round_strategy` transitions succeed but surface warnings in command output.
- `submit-round` may emit `next_round_hint`, but it does not persist next-round state.
- `summary.md` is no longer the place for full strategy-state history.
- no new JSON archive is introduced for strategy-state history.
