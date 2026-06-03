---
title: Optimize Check Loop
created: 2026-04-14
summary: Simplify optimize orchestration by moving baseline and round validation into a reusable skill-backed check flow, while keeping supervisor as an optional metadata-only audit layer.
---

# Optimize Check Loop Design

## Summary

- Add one dedicated `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill for baseline and round validation.
- Require optimize workers to call that skill themselves after baseline preparation and after each completed round.
- Keep `--supervisor off` as a long-running worker-owned optimize session.
- Keep `--supervisor on` as a worker-per-round loop where supervisor decides whether another worker round should run.
- Restrict supervisor repairs to metadata, briefs, summaries, and other non-core artifacts derived from existing facts.
- Keep CLI/runtime orchestration thin and move optimize workflow checks back into skill-defined behavior.

## Goals

- Ensure every optimize run validates baseline state before the first optimization round is accepted.
- Ensure every optimization round is checked before the agent may move on to the next round.
- Make the validation contract reusable by workers, supervisors, and CLI helpers without duplicating rule logic.
- Keep optimize workflow behavior centered in skills and helper scripts instead of scattering it across runtime code.
- Clarify the behavioral difference between supervised and unsupervised optimize runs.

## Non-Goals

- Do not make the supervisor responsible for kernel or operator implementation changes.
- Do not move optimize-domain reasoning out of skills and into the CLI.
- Do not require the runtime to infer round validity from free-form agent output.
- Do not introduce a new parallel optimize search model in this change.
- Do not require the CLI to micromanage every baseline or round repair step.

## Problem

- The current supervised optimize flow mixes several responsibilities:
  - worker execution
  - supervisor execution
  - runtime-side artifact gating
  - round-to-round continuation logic
- This makes the control loop harder to reason about and still leaves room for a code agent to continue optimizing before a clean validation handoff happens.
- The optimize workflow already belongs primarily in skills, but the current gate model puts too much workflow-specific checking burden on runtime code.
- We want a cleaner model where the agent itself performs the required checks as part of the workflow, and where supervision remains optional and clearly scoped.

## Design Principles

- Treat the optimize skill as the source of truth for how workers progress through baseline preparation and optimization rounds.
- Treat `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` as the source of truth for whether a baseline or round is technically acceptable.
- Treat the supervisor as an audit layer, not as the primary technical validator.
- Treat runtime as an orchestration layer that launches the right role at the right time, but does not own optimize-specific technical policy.

## High-Level Workflow

The optimize flow has three logical phases:

1. Baseline establishment
2. Round execution and validation
3. Optional supervisor audit between rounds

Each phase has one primary owner:

- `worker` owns baseline preparation and round execution
- `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` owns baseline and round validation
- `supervisor` owns optional audit, metadata-only repair, and continue-or-stop decisions in supervised mode

## `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` Skill

Add one dedicated skill, `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`, with one bundled helper script.

That script exposes two subcommands:

```bash
python optimize_check.py check-baseline --baseline-dir <path>
python optimize_check.py check-round --round-dir <path>
```

### Responsibilities

`check-baseline` validates whether the canonical optimize baseline is complete and reusable.

`check-round` validates whether one completed optimize round is acceptable before the session may continue.

Both checks should produce:

- a machine-readable result for programmatic reuse
- a short human-readable summary that the code agent can use for repair

### Behavioral Intent

The purpose of this skill is not to continue optimization work.

Its purpose is to answer:

- Is the baseline acceptable?
- Is this round acceptable?
- If not, what specifically must be repaired before the workflow may proceed?

## Worker Contract

The optimize worker contract changes as follows.

### Baseline

The worker is responsible for deciding whether baseline preparation is needed.

If baseline artifacts are missing or incomplete, the worker must:

1. prepare or repair the baseline
2. run `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round check-baseline`
3. keep repairing the baseline until the check passes

The worker must not begin the first optimization round until baseline validation passes.

### Rounds

After finishing an optimization round, the worker must:

1. run `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round check-round` against the current round directory
2. inspect the returned issues
3. repair the current round if the check fails
4. repeat until the round passes

The worker must not begin the next optimization round until the current round passes `check-round`.

This rule applies in both supervised and unsupervised optimize flows.

## Mode Semantics

### `--supervisor off`

In unsupervised mode, one long-running worker owns the entire optimize session.

That worker must:

- establish and validate baseline when needed
- execute optimization rounds
- validate each round before proceeding
- continue until its own stop condition is met and any CLI-level round minimum is satisfied

In this mode, the runtime should not insert a supervisor loop.

The optimize skill itself should emphasize that the code agent owns the full session and must keep making progress until the session should stop.

### `--supervisor on`

In supervised mode, each worker invocation owns only one optimization round.

That worker invocation must still:

- establish and validate baseline when needed
- complete one optimization round
- validate that round before exiting

After the worker exits, runtime launches the supervisor.

The supervisor then decides whether the optimize session should continue or stop.

If continuation is approved, runtime launches a fresh worker for the next round.

In this mode, the optimize skill should emphasize that the code agent owns exactly one round, because the outer loop belongs to supervised orchestration.

## Supervisor Contract

Supervisor remains optional and only applies when `--supervisor on`.

Its responsibilities are:

- audit the completed round after worker-side validation
- review `opt-note.md`, round summaries, briefs, and recorded evidence
- repair non-core artifacts when the repair is derived from existing facts
- produce the next-round brief when continuation is appropriate
- decide whether the supervised session should continue or stop

Supervisor must not:

- modify the operator or kernel implementation
- perform a new optimize round
- fabricate missing correctness, benchmark, profile, or IR evidence
- convert missing technical work into a metadata-only pass

### Allowed Supervisor Repairs

Examples of allowed repairs:

- rewriting or normalizing `summary.md`
- refreshing a round handoff brief
- repairing or clarifying session metadata
- updating `opt-note.md` from already-established facts

### Disallowed Supervisor Repairs

Examples of disallowed repairs:

- changing Triton code
- making new optimization edits
- adding invented benchmark conclusions
- pretending that missing evidence exists

## Runtime Responsibilities

Runtime should become smaller and more explicit.

### In Both Modes

Runtime remains responsible for:

- preparing the workspace and staging skills
- building the correct worker prompt for the selected mode
- launching the requested backend
- handling outer orchestration concerns such as process lifecycle, retries, cleanup, and CLI-level options

Runtime should not become the primary owner of baseline or round validation rules.

### In `--supervisor off`

Runtime launches one optimize worker and lets that worker run the session.

The worker itself owns baseline and round checking through `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`.

### In `--supervisor on`

Runtime becomes a simple outer loop:

1. launch one worker invocation
2. launch one supervisor invocation
3. read the supervisor decision
4. either stop or launch the next worker invocation

Runtime should not independently replicate the detailed baseline or round gate logic when the same rules already live in `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`.

## Prompt And Guidance Changes

The prompt model should clearly distinguish the two optimize modes.

### Worker Prompt In `--supervisor off`

The prompt should emphasize:

- this is a full optimize session
- baseline must be validated before the first round
- every round must pass `check-round` before the next round may start
- the agent should continue optimizing until the session should stop

### Worker Prompt In `--supervisor on`

The prompt should emphasize:

- this invocation owns exactly one optimization round
- baseline must be validated if baseline work is needed
- the completed round must pass `check-round` before the invocation ends
- the agent must not decide the outer session loop by itself

### Supervisor Prompt

The supervisor prompt should emphasize:

- audit and handoff only
- metadata-only repair scope
- no operator implementation edits
- continue-or-stop decision making based on existing facts

## Reuse Of Checks

The same `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` script should be reusable from multiple places:

- the optimize worker skill
- the optimize supervisor skill
- any focused runtime or test helper that needs deterministic validation

This keeps baseline and round validation rules in one place.

## Error Handling

When baseline or round checks fail, the default workflow should be repair-and-recheck, not immediate session termination.

Hard failure is reserved for cases where:

- the check cannot run
- artifacts are irreparably inconsistent
- the underlying optimize run failed in a way that prevents meaningful repair

In normal cases, failed checks should produce actionable issues that are fed back to the worker.

## Documentation Impact

Update:

- `README.md`
  - explain `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` as part of optimize workflow expectations
  - document the behavioral difference between supervised and unsupervised optimize
- optimize skill docs
  - require baseline and round checks
- optimize supervisor skill docs
  - limit supervisor to metadata-only repair

Update `AGENTS.md` only if the project wants these semantics called out as durable repository policy.

## Testing

Add coverage for:

- `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round check-baseline` pass and failure cases
- `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round check-round` pass and failure cases
- unsupervised worker prompts requiring baseline and round checks
- supervised worker prompts requiring one-round ownership plus mandatory checks
- supervisor prompts enforcing metadata-only repair boundaries
- supervised runtime looping on supervisor decisions without re-implementing the full check policy

## Recommendation

Adopt `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` as the shared baseline and round validation layer, require workers to use it directly, and keep supervisor as an optional metadata-only audit layer.

This design simplifies orchestration, keeps workflow behavior centered in skills, and makes the two optimize modes easier to understand and maintain.
