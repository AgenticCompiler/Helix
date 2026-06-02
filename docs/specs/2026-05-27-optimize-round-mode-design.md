# Optimize Round Mode Design

## Summary

- Replace `--supervise/--supervisor` with `--round-mode {continuous,checked,supervised}` on `optimize` and `optimize-batch`.
- Treat `continuous` as the long-running optimize session mode where one agent owns multiple rounds and decides when to stop.
- Treat `checked` as a one-round-per-invocation mode where the CLI validates each finished round and decides whether to continue, stop, or fail.
- Treat `supervised` as a one-round-per-invocation mode where the CLI validates each finished round first, then a supervisor audit pass decides whether to continue, stop, or fail.
- Add a CLI-owned baseline preflight before the round loop so `checked` and `supervised` can keep round invocations focused on exactly one optimization round.
- Keep `triton-npu-optimize-check` as the shared technical validation authority for baseline and round acceptance.

## Goals

- Make optimize mode selection explicit and understandable from the command line.
- Introduce a middle mode where each agent invocation completes one round without requiring supervisor involvement.
- Keep `checked` and `supervised` round loops technically consistent by reusing one CLI-owned validation gate.
- Remove baseline ambiguity from round-scoped invocations by validating or repairing the baseline before the round loop starts.
- Preserve the current artifact-driven optimize workflow and round records while simplifying round-to-round context flow.

## Non-Goals

- Do not preserve `--supervise` or `--supervisor` as compatibility aliases in this change.
- Do not move optimize-domain reasoning out of skills and into Python runtime code.
- Do not redefine the artifact schema for `baseline/`, `opt-round-N/`, `round-state.json`, `attempts.md`, or `summary.md`.
- Do not make the supervisor responsible for basic round completeness or baseline acceptance.
- Do not change the meaning of resume modes such as `--resume auto|continue|fresh`.

## Problem

The current optimize CLI exposes a two-state supervised toggle:

- one long-running optimize session with no supervisor loop
- one worker-per-round loop with a supervisor pass after every round

That model leaves a gap between the two existing choices:

- some users want a bounded one-round-per-invocation flow
- they still want the CLI to validate round artifacts and decide whether another round should run
- they do not want a supervisor audit pass on every round

The current prompt and runtime split also mixes baseline handling into round-scoped behavior. That weakens the meaning of "one invocation owns one round" because a round invocation may still need to spend time preparing the baseline before the round actually starts.

## User-Facing Design

### CLI Flag

Add a new optimize-only flag:

```bash
--round-mode {continuous,checked,supervised}
```

Default:

- `optimize`: `--round-mode continuous`
- `optimize-batch`: `--round-mode continuous`

### Mode Meanings

`continuous`

- One optimize agent owns the full session.
- The agent may complete multiple rounds in one invocation.
- The agent validates baseline and round state itself through `triton-npu-optimize-check`.
- The agent decides when the optimize session should stop.

`checked`

- One optimize agent invocation owns exactly one optimization round.
- The CLI validates the finished round after the agent exits.
- The CLI decides whether to continue, stop, or fail.
- No supervisor audit pass runs.

`supervised`

- One optimize agent invocation owns exactly one optimization round.
- The CLI validates the finished round after the agent exits.
- If the round passes technical validation, a supervisor audit pass runs.
- The supervisor decides whether to continue, stop, or fail.

### Examples

```bash
uv run triton-agent optimize --input operator.py
uv run triton-agent optimize --input operator.py --round-mode checked
uv run triton-agent optimize --input operator.py --round-mode supervised
uv run triton-agent optimize-batch --input operators_root --round-mode checked
```

## Baseline Preflight

Before the CLI launches the first optimize agent invocation, it should run a baseline preflight through `triton_agent.optimize.checks.check_baseline(...)`.

The preflight result should be normalized into three runtime states:

- `ready`: a valid reusable baseline already exists
- `needs_prepare`: the baseline is missing
- `needs_repair`: the baseline exists but fails the baseline contract

### `continuous`

In `continuous` mode, baseline preflight exists only to sharpen the prompt:

- when the baseline is `ready`, the prompt should explicitly tell the agent to reuse the validated baseline and avoid rebuilding it
- when the baseline is `needs_prepare` or `needs_repair`, the prompt should explicitly tell the agent to repair the baseline before opening round 1

The agent still owns baseline repair in this mode.

### `checked` and `supervised`

In `checked` and `supervised`, baseline preflight becomes a distinct pre-round stage:

1. the CLI checks whether `baseline/` is valid
2. if it is not valid, the CLI launches one baseline-focused optimize invocation
3. the CLI reruns `check_baseline(...)`
4. only after the baseline passes does the CLI enter the one-round loop

This keeps the round-scoped invocations honest: once the checked or supervised round loop begins, each optimize invocation owns exactly one round rather than a mixture of baseline setup and round work.

## Round-To-Round Context

The CLI owns round-to-round continuation context for `checked` and `supervised`.

`checked`

- the CLI runs technical validation after each round
- the CLI decides continue, stop, or fail
- when another round is needed, the CLI injects the previous round's validation result directly into the next worker prompt
- no live handoff file is required

`supervised`

- the CLI runs technical validation first
- the supervisor still writes `.triton-agent/supervisor-report.md`
- when another round is needed, the CLI reads `supervisor-report.md` and injects its content into the next worker prompt
- the worker does not read a separate `.triton-agent/round-brief.md`

This removes the extra "CLI writes a handoff file, then the worker reads that file" protocol. The worker only receives the current invocation prompt.

## Round Semantics

### `continuous`

The runtime launches one long-running optimize agent session.

That agent owns:

- baseline preparation or repair when needed
- repeated optimization rounds
- round-local `check-round` execution
- deciding when the session should stop

The CLI does not insert a round-to-round gate and does not launch a supervisor.

### `checked`

The runtime launches one optimize agent per round.

Each invocation owns:

- making one coherent optimization attempt
- producing the required `opt-round-N/` artifacts
- recording `round-state.json`, including `round_disposition`

After the invocation exits, the CLI runs the shared technical round gate.

The CLI then decides whether to:

- continue with a fresh next-round invocation
- stop the optimize session
- fail the optimize session

### `supervised`

The runtime also launches one optimize agent per round.

The CLI runs the same technical round gate as `checked`.

Only after technical validation passes does the runtime launch a supervisor audit pass.

The supervisor then decides whether to:

- continue with a fresh next-round invocation
- stop the optimize session
- fail the optimize session

This keeps supervisor focused on audit and handoff rather than basic technical acceptance.

## Runtime Architecture

### Request Model

Replace the existing optimize supervision flag with:

```python
round_mode: Literal["continuous", "checked", "supervised"]
```

This field belongs on optimize-specific option and request models.

Keep `optimize_role` as an internal execution field:

- unset in `continuous`
- `"worker"` for the round agent in `checked` and `supervised`
- `"supervisor"` for the audit pass in `supervised`

### Top-Level Dispatch

`run_optimize_request(...)` should dispatch to two runtime families:

- `execute_continuous_optimize(...)`
- `execute_multi_invocation_optimize(...)`

`execute_multi_invocation_optimize(...)` should handle both `checked` and `supervised`.

This preserves shared ownership for:

- skill staging
- resume resolution
- output rendering
- session logging
- cleanup

### Shared CLI Technical Gate

Add one CLI-owned helper that performs round acceptance after a round agent exits.

Responsibilities:

1. resolve the latest `opt-round-*` directory
2. run `triton_agent.optimize.checks.check_round(...)`
3. interpret the returned decision:
   - `pass` -> accepted
   - `revise-required` -> repair required
   - `hard-fail` -> hard failure
4. when the round passes, read `round-state.json`
5. combine `round_disposition` with `min_rounds` to produce a runtime gate decision

The technical gate should reuse the existing `GateDecision` values where they fit:

- `PASS_CONTINUE`
- `PASS_STOP`
- `REVISE_REQUIRED`
- `HARD_FAIL`

`REVISE_METADATA` should remain reserved for supervisor-specific metadata repair paths rather than being emitted by the CLI technical gate.

### Repair Loop Semantics

If the CLI technical gate returns `REVISE_REQUIRED`, the runtime should not fail immediately.

Instead it should:

1. build a repair-focused continuation prompt
2. relaunch the optimize agent for the current round
3. rerun the technical gate after that invocation completes

Only `HARD_FAIL` should terminate the optimize session immediately.

This keeps `checked` and `supervised` aligned with the existing optimize repair model where incomplete or invalid round artifacts trigger another bounded repair pass rather than an immediate fatal exit.

### `checked` Versus `supervised`

After a round passes the technical gate:

- `checked` uses the CLI gate decision directly
- `supervised` launches a supervisor audit pass, and the supervisor becomes the final authority on whether the session continues or stops

The runtime therefore becomes:

`checked`

1. run round agent
2. run CLI technical gate
3. continue, stop, or fail

`supervised`

1. run round agent
2. run CLI technical gate
3. if technically valid, run supervisor audit
4. continue, stop, or fail

## Prompt And Guidance Design

### `continuous` Prompt

Keep the current long-running optimize-session contract:

- this invocation owns the end-to-end optimize session
- when the baseline is invalid, repair it before starting round 1
- use `check-round` after each completed round
- decide whether the session should continue or stop based on evidence and `check-round` output

The wording should be updated to use `continuous` terminology rather than `supervise off` terminology.

### `checked` Round Prompt

The checked round prompt should say:

- this invocation owns exactly one round
- baseline has already been validated before the round loop begins
- complete the current round and produce all required round artifacts
- record `round-state.json`, including `round_disposition`
- do not self-approve whether the session should continue
- the CLI will validate the round after this invocation exits
- if the round needs repairs, a later invocation will return with a repair brief

The checked round prompt should not require the agent to run `check-round` itself before exiting.

### `supervised` Round Prompt

The supervised round prompt should share the same round-agent core contract as `checked`.

It should additionally note that:

- after the CLI validates the round, a supervisor audit pass will review the result

The supervised round prompt should also stop requiring the round agent to run `check-round` itself before exit.

### Supervisor Prompt

The supervisor prompt should assume that the round has already passed CLI technical validation.

Its responsibilities should narrow to:

- auditing whether the round conclusion matches existing evidence
- checking whether the recorded analysis level and evidence path are coherent
- performing metadata-only repair derived from existing facts
- writing the next-round brief or stop brief
- deciding whether the session should continue, stop, or require another repair-focused round

`REVISE_REQUIRED` from the supervisor should now mean:

- technical artifacts passed the CLI gate
- but the audited round still lacks sufficient analysis, explanation, or evidence integrity for the next step

It should no longer mean simple artifact incompleteness.

### Shared Guidance Files

`continuous`

- keep the current self-contained optimize guidance file approach

`checked`

- use a shared round-gated guidance file
- include `.triton-agent/round-brief.md` as the live handoff file
- do not mention supervisor-specific runtime files

`supervised`

- use the same shared round-gated guidance baseline
- include both `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md`

The checked shared guidance should not mention "worker and supervisor roles" because that would overfit the checked mode to a supervisor-driven model it does not use.

## Artifact Contract Refinement

`round_disposition` should remain required in `round-state.json`, but its meaning should be narrowed.

Old interpretation:

- the round author decides whether the optimize session should continue or stop

New interpretation:

- the round author records a recommendation based on current evidence about whether another round appears justified

Authority by mode:

- `continuous`: the long-running agent may act directly on that recommendation
- `checked`: the CLI uses that recommendation after technical validation, subject to `min_rounds`
- `supervised`: the supervisor may accept or override that recommendation

This keeps the field useful without incorrectly treating a round-scoped invocation as the final session controller in `checked` or `supervised`.

## Session Artifacts

### `continuous`

Keep the current unsupervised session artifact shape:

- one self-contained optimize guidance file
- archive/session logging

### `checked`

Prepare a round-gated artifact set containing:

- shared guidance
- `.triton-agent/round-brief.md`
- archive/session logging

Do not create a live `supervisor-report.md` in checked mode.

### `supervised`

Prepare the round-gated artifact set plus supervisor-specific live files:

- shared guidance
- `.triton-agent/round-brief.md`
- `.triton-agent/supervisor-report.md`
- archive/session logging
- supervisor history snapshots

This prevents checked mode from carrying a misleading supervisor file contract.

## `--interact` Semantics

`continuous`

- preserve current interactive optimize behavior as one long-running interactive agent session

`checked`

- each round agent invocation may run interactively
- the CLI technical gate always runs non-interactively as Python runtime logic

`supervised`

- each round agent invocation may run interactively
- the supervisor audit pass remains non-interactive

## `--min-rounds` Semantics

`continuous`

- preserve the existing meaning where the long-running agent uses `check-round --min-rounds ...` guidance to know whether stopping is allowed

`checked`

- the CLI technical gate becomes the authority that combines `round_disposition` with `min_rounds`
- a checked round may recommend stop, but the CLI must force continuation until the minimum round requirement is satisfied

`supervised`

- the CLI technical gate should still reject early stop before the minimum round requirement is satisfied
- once the minimum is satisfied, the supervisor remains the final authority on continue versus stop

## Batch Behavior

`optimize-batch` should accept the same `--round-mode` flag and apply it independently per workspace.

- `continuous`: each workspace uses one long-running agent session
- `checked`: each workspace uses a CLI-gated one-round loop
- `supervised`: each workspace uses a CLI-gated one-round loop plus supervisor audit

Batch status handling and concurrency behavior remain otherwise unchanged.

## Documentation Changes

Update:

- `README.md`
  - replace `--supervise/--supervisor` documentation with `--round-mode`
  - explain all three mode meanings
  - document baseline preflight behavior for checked and supervised modes
- optimize design notes that still describe `supervise on/off` as the public API
- any prompt or runtime comments that describe checked or supervised loops using old supervision terminology

## Testing

Add or update tests for:

- parser accepts `--round-mode continuous|checked|supervised`
- parser default is `continuous`
- optimize request models map the selected round mode correctly
- `continuous` dispatches to the continuous runtime path
- `checked` dispatches to the round-gated runtime without supervisor execution
- `supervised` dispatches to the round-gated runtime with supervisor execution after technical pass
- baseline preflight:
  - ready baseline reuses the baseline and skips the baseline-focused pre-phase
  - missing baseline triggers the baseline-focused pre-phase
  - invalid baseline triggers the baseline repair pre-phase
- checked technical gate:
  - pass + round recommendation continue => relaunch next round
  - pass + round recommendation stop + min rounds satisfied => stop
  - pass + round recommendation stop + min rounds unsatisfied => continue
  - revise-required => write repair brief and relaunch repair pass
  - hard-fail => fail immediately
- supervised technical gate:
  - technically invalid round never launches supervisor
  - technically valid round launches supervisor
  - supervisor may still return `revise-required`
- checked artifacts do not create or require a live `supervisor-report.md`
- supervised artifacts still archive the final supervisor report and history snapshots

## Recommendation

Adopt `--round-mode {continuous,checked,supervised}` and implement the new middle mode by sharing one CLI-owned technical gate between `checked` and `supervised`.

The key design choices are:

- baseline becomes a CLI-owned preflight and pre-phase for round-gated modes
- round-scoped invocations stop owning final `check-round` acceptance
- the CLI becomes the technical validator in `checked` and `supervised`
- the supervisor narrows to audit and handoff after technical validity is already established

This gives optimize three modes with clear semantics instead of one overloaded supervise toggle:

- `continuous`: agent-owned session control
- `checked`: CLI-owned round control
- `supervised`: CLI technical control plus supervisor audit control
