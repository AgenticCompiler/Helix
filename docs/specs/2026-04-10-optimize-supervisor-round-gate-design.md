# Optimize Supervisor Round Gate Design

> **Superseded note:** The current implementation no longer creates `.triton-agent/roles/*` files and no longer uses a dedicated `optimize-supervisor` skill. Supervisor behavior now comes from the launch prompt plus `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md`.

## Summary

- Split `optimize` into explicit worker rounds instead of one unconstrained long-running agent session.
- Add a supervisor gate after every round to enforce the optimize skill workflow before the next round may start.
- Keep optimization work in the worker role; keep audit, metadata repair, and next-step guidance in the supervisor role.
- Introduce a small structured round contract so the CLI can evaluate compliance without guessing from free-form agent output.
- Keep the local `skills/` directory as the workflow source of truth while separating shared workspace guidance from role-specific instructions.

## Goals

- Prevent optimize agents from skipping required workflow steps such as hypothesis recording, comparable perf capture, `summary.md`, and `opt-note.md` updates.
- Let the CLI enforce round boundaries and workflow gates instead of relying only on prompt wording.
- Allow limited safe supervisor repair of missing metadata when the underlying evidence already exists.
- Produce clear next-round handoff guidance so later rounds continue from the approved state instead of re-exploring from scratch.
- Preserve the existing separation between CLI orchestration and skill-defined workflow semantics.

## Non-Goals

- Do not move optimize-domain reasoning out of the optimize skill and into the CLI.
- Do not let the supervisor invent missing benchmark, profiler, or IR evidence.
- Do not introduce parallel multi-branch optimize search in this first design.
- Do not require the supervisor to become a second free-form optimize agent.
- Do not replace `attempts.md`, `summary.md`, or `opt-note.md` with structured state alone.

## Problem

- The current optimize flow launches one code agent with staged skills plus a temporary optimize guidance file and expects the agent to follow the workflow on its own.
- Existing orchestration only supervises liveness concerns such as stalled runs and minimum round count, as implemented in [supervisor.py](/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/supervisor.py).
- The optimize skill and guidance already require concrete artifacts and workflow steps, but the CLI does not currently gate on those requirements.
- As a result, agents may:
  - skip or under-specify round hypotheses and supporting evidence
  - fail to produce comparable perf data
  - omit or underfill `summary.md`
  - update `opt-note.md` inconsistently
  - continue optimizing after a workflow-violating round instead of first repairing the session state

## High-Level Architecture

The optimize command becomes an explicit orchestration loop:

1. Prepare workspace skills and shared optimize guidance.
2. Launch a fresh worker agent for one round.
3. Run a fresh supervisor agent against the completed round.
4. Apply the supervisor decision.
5. Either stop, restart the worker with a repair brief, or launch the next worker round with a new brief.

### Roles

- `worker`
  - Owns one optimization round.
  - May edit the round-local operator copy and produce round artifacts.
  - Does not decide whether workflow compliance is sufficient for session continuation.
- `supervisor`
  - Audits the just-completed round.
  - May normalize or repair metadata only when the underlying evidence already exists.
  - Produces the gate decision and next-round brief.
  - Must not perform open-ended optimization work.

### Invocation Model

- Every worker round is a fresh agent invocation.
- Every supervisor gate is a fresh agent invocation.
- Do not reuse one agent session across worker and supervisor roles.

This avoids role confusion, stale instruction carry-over, and backend-specific session state leaking worker behavior into supervisor behavior.

## Round Contract

Each completed round must leave a machine-checkable package under `opt-round-N/`.

### Required Round Artifacts

- `opt-round-N/<optimized-operator>`
- `opt-round-N/attempts.md`
- `opt-round-N/summary.md`
- `opt-round-N/perf.txt` or the existing standard comparable perf artifact
- `opt-round-N/round-state.json`

Optional artifacts remain:

- `opt-round-N/profile/`
- `opt-round-N/ir/`

The workspace must also contain an updated `opt-note.md` after each completed round.

### `round-state.json`

Add a minimal structured file so the supervisor and CLI can validate round completion without inferring everything from Markdown.

Required fields:

- `round`
- `parent_round`
- `hypothesis`
- `evidence_sources`
- `correctness_status`
- `benchmark_status`
- `perf_artifact`
- `summary_path`
- `opt_note_updated`
- `next_recommendation`

Optional fields:

- `analysis_skipped_reason`
- `profile_dir`
- `ir_dir`
- `validated_candidate`

The structured file does not replace human-readable notes. It exists to make supervisor gating deterministic and to cross-check agent-written summaries against concrete artifacts.

## Supervisor Gate

The supervisor evaluates each round in a fixed order. Earlier failures block later checks.

### 1. Structure Integrity

Verify:

- the round directory exists
- round-local operator output exists
- `attempts.md` exists
- `summary.md` exists
- `round-state.json` exists and is parseable
- `opt-note.md` was updated
- the parent round is identifiable

Missing metadata may be eligible for supervisor repair only when the missing content can be reconstructed from existing round facts.

### 2. Evidence Integrity

Verify:

- the round hypothesis is recorded before the code-changing outcome is summarized
- supporting evidence is explicitly named
- if profiling or IR capture is absent, the round explains why existing evidence was sufficient
- correctness evidence exists
- benchmark evidence exists
- perf data is comparable to the expected baseline or parent cases

Missing correctness, benchmark, or comparable perf evidence is never metadata-only repair.

### 3. Result Validity

Verify:

- correctness passed
- correctness was completed before benchmark acceptance
- claimed performance outcome matches concrete perf data
- `summary.md` conclusions agree with `round-state.json` and perf artifacts
- any promoted winner or validated branch is justified by the recorded metrics

### 4. Session Continuity

Verify:

- `opt-note.md` remains internally consistent
- current best and validated branch markers are not corrupted by a failed round
- the next parent round is explicit
- the next step is clear enough to hand to another worker

### Gate Decisions

The supervisor emits one of five decisions:

- `pass-continue`
  - The round is valid and the session should continue with a next-round brief.
- `pass-stop`
  - The round is valid and the optimize session may stop.
- `revise-metadata`
  - The round is substantively valid but metadata repair is required before continuation.
- `revise-required`
  - The round is missing required evidence or workflow state and must be repaired by another worker run before the session may continue.
- `hard-fail`
  - The round is invalid in a way that should stop the optimize session immediately.

## Repair Policy

### Allowed Supervisor Repairs

The supervisor may repair only presentation and bookkeeping derived from existing facts, for example:

- normalize `summary.md` wording from existing benchmark or profiler results
- patch or append the round entry in `opt-note.md`
- refresh the `## Overall Summary` block from existing validated round data
- normalize perf artifact references
- generate a clearer next-round brief from already-recorded evidence

### Disallowed Supervisor Repairs

The supervisor must not:

- fabricate missing benchmark, profiler, or IR evidence
- claim a round passed correctness without real validation output
- infer comparable perf data from non-comparable measurements
- silently promote a failed or unvalidated round to current best
- take over open-ended operator optimization

If the session needs new experiments, the supervisor must block and hand the task back to a worker round.

## Guidance And Skill Model

The current optimize runtime stages repository skills and writes a temporary top-level guidance file for the optimize agent. That remains appropriate, but the guidance model must distinguish shared workspace rules from role-specific behavior.

### Shared Guidance

Keep one shared temporary workspace guidance file such as `AGENTS.md` or `CLAUDE.md` that applies to all optimize-role agents in the session. It should define:

- this workspace is under optimize orchestration
- staged skills are the workflow source of truth
- every launched agent must read its role brief before acting
- supervisor repair limits and non-goals

The shared guidance must stay role-neutral. Do not put worker-only or supervisor-only instructions in the top-level guidance file, because backend memory-file behavior may lift that file into higher-priority prompt context. In particular, the shared guidance must not say things like "improve the operator" or "this invocation is an audit pass."

### Role Briefs

Add separate role-specific documents under a temporary orchestration area, for example:

- `.triton-agent/roles/optimize-worker.md`
- `.triton-agent/roles/optimize-supervisor.md`
- `.triton-agent/round-brief.md`
- `.triton-agent/supervisor-report.md`

The worker prompt must point to the worker role brief and current round brief.
The supervisor prompt must point to the supervisor role brief and current round artifacts.

Role identity must come from the launch prompt, not from the shared guidance file. The role briefs expand role behavior, but they should not be the only place where role assignment is declared.

### Skills

Stage skills once for the whole optimize orchestration as today.

Recommended role-to-skill mapping:

- worker uses `optimize`
- supervisor uses a new `optimize-supervisor` skill or equivalent audit-focused skill

The supervisor skill may reference the optimize skill as read-only workflow context, but it must remain audit-oriented rather than becoming a second optimize skill.

## Prompt Contract

### Worker Prompt

The worker prompt should explicitly say:

- this invocation is the optimize worker role
- this invocation owns exactly one round
- read the worker role brief and round brief first
- produce all required round artifacts
- stop after completing one auditable round
- do not self-approve session continuation

### Supervisor Prompt

The supervisor prompt should explicitly say:

- this invocation is the optimize supervisor role
- this invocation is an audit and handoff pass, not a new optimization round
- read the supervisor role brief and the completed round artifacts first
- apply only allowed metadata repairs
- emit a structured gate result
- produce a concrete next-round brief when continuation is allowed

## CLI Contract

The optimize orchestration layer becomes responsible for the round loop.

### Required CLI Behavior

- start optimize by preparing shared skills and shared guidance
- launch a worker round
- always launch a supervisor gate after worker completion
- refuse to start the next round unless the supervisor returns `pass-continue` or a successfully repaired `revise-metadata`
- relaunch a worker repair pass when the supervisor returns `revise-required`
- stop immediately on `hard-fail`
- stop successfully on `pass-stop`

### Gate Output

The supervisor should emit a structured result such as `gate-result.json` with:

- `decision`
- `blocking_issues`
- `auto_repairs_applied`
- `next_parent_round`
- `next_hypothesis`
- `required_evidence_for_next_round`

The CLI should use this structured output as the source of truth and treat accompanying Markdown as operator-facing explanation.

## Implementation Shape

Likely implementation areas:

- [src/triton_agent/optimize/supervisor.py](/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/supervisor.py)
  - expand from stall/min-round recovery into round gate orchestration
- [src/triton_agent/prompts.py](/Users/cdj/Projects/triton-agent/src/triton_agent/prompts.py)
  - split worker and supervisor prompt builders
- [src/triton_agent/optimize/guidance.py](/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/guidance.py)
  - render shared guidance plus role briefs instead of one worker-only guidance file
- `src/triton_agent/optimize/`
  - add round contract parsing, gate result models, and artifact inspection helpers
- `skills/`
  - add an audit-oriented optimize supervisor skill

## Phase Plan

### Phase 1: MVP Gate

- Keep worker rounds and supervisor rounds as fresh invocations.
- Add `round-state.json`.
- Add gate-result parsing and fixed gate decisions.
- Support metadata-only repair.
- Do not let the supervisor rerun experiments.

### Phase 2: Controlled Recovery

- Allow narrowly scoped reruns of existing harnesses when explicitly requested by the gate design.
- Improve automatic summary and `opt-note.md` repair from existing facts.
- Strengthen next-round brief synthesis.

## Testing

- Unit tests for round artifact inspection and `round-state.json` parsing
- Supervisor tests for each gate decision path
- Prompt tests for worker versus supervisor role wording
- Optimize runtime tests covering worker-to-supervisor orchestration
- Guidance tests covering shared guidance plus role briefs
- Verification with:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`

## Open Questions

- Whether the supervisor skill should be a completely separate skill or an audit-oriented appendix to the existing optimize skill
- How much metadata repair should be done by the supervisor agent versus deterministic Python helpers
- Whether `revise-required` should relaunch the same round number or require a new repair round identifier
