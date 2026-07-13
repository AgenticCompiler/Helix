# Optimize Batched Round Mode Design

## Summary

- Remove `--round-mode continuous` from the optimize CLI surface and keep only `checked` and `supervised`.
- Add `--round-batch-size` with default `10` so one worker invocation is asked to complete multiple rounds before the CLI restarts it.
- Reframe `checked` as the default batched optimize mode: the CLI validates every new round after each worker batch and feeds the repair or continuation summary into the next worker launch.
- Reframe `supervised` as batched worker execution plus one supervisor audit per completed batch.
- Replace the round checker's session-level `--min-rounds` contract with `--current-round` and `--final-round` so stopping policy stays in the CLI batch loop instead of in a per-round checker.

## Problem

- The current optimize runtime has two different execution models:
  - `continuous` launches one long-running worker session and resumes it until enough `opt-round-*` directories exist.
  - `checked` and `supervised` launch one worker per round, then optionally one supervisor per round.
- The desired new behavior is different from both:
  - the optimize session still has a global `--min-rounds` requirement
  - each worker launch should own multiple rounds in sequence
  - after at most `--round-batch-size` rounds, the CLI should restart the worker
  - `supervised` should audit once per batch, not once per round
- The current round checker also mixes two concerns:
  - validating whether the current round is structurally and semantically acceptable
  - deciding whether the overall optimize session may stop based on `--min-rounds`
- That coupling makes batch-oriented orchestration awkward because a per-round checker should not be the authority for session-level stop policy.

## Goals

- Keep `--min-rounds` as the optimize-session stopping requirement.
- Add `--round-batch-size` as the worker restart boundary for both `checked` and `supervised`.
- Make `checked` the default non-supervised optimize mode.
- Ensure batch validation checks every newly completed round, not just the latest round.
- Preserve the existing repair-loop behavior where CLI-generated warnings and workflow issues are passed into a later worker invocation.
- Make `supervised` run exactly one supervisor pass per completed batch, with `--round-batch-size 1` naturally degenerating to the current one-round-per-supervisor behavior.
- Update prompts so batched workers know their current target range and are explicitly told not to pre-plan all rounds in advance.

## Non-Goals

- Do not move optimize-domain reasoning out of the staged skills and into the CLI.
- Do not add a separate third round mode for batched execution.
- Do not introduce parallel round execution inside a batch.
- Do not make the supervisor perform open-ended optimization or generate missing benchmark, correctness, profiling, or IR evidence.
- Do not make the round checker responsible for deciding global session completion from workspace state alone.

## User-Facing Behavior

### CLI Surface

- `--round-mode` becomes `checked | supervised`.
- The default `--round-mode` becomes `checked`.
- `--round-batch-size` is added for `optimize` and `optimize-batch`.
- `--round-batch-size` defaults to `10` and must be at least `1`.
- `--min-rounds` keeps its current meaning and remains required to be at least `1`.

### Mode Semantics

- `checked`
  - Launch a worker for a batch of rounds.
  - After the worker exits, the CLI checks every newly created round in that batch.
  - If the batch is valid but the global minimum round count is still unmet, the CLI launches the next worker batch with a continuation summary.
- `supervised`
  - Same worker behavior as `checked`.
  - After the CLI finishes checking the completed batch, launch one supervisor invocation for that batch.
  - Feed the supervisor report into the next worker batch or use it to stop with failure.

### Interactive Mode

- `--interact` should no longer be accepted for optimize once `continuous` is removed from the command surface.
- The error should clearly say that interactive optimize is not supported by the batched checked/supervised flow.

## High-Level Design

### One Batch, One Intended Worker Lifecycle

The CLI should treat each batch as the primary worker lifecycle boundary.

- Let `completed_rounds` be the number of accepted rounds already in the workspace.
- Compute the next batch target as:
  - `batch_start = completed_rounds + 1`
  - `batch_end = min(completed_rounds + round_batch_size, min_rounds)`
- Launch a worker prompt that instructs the agent to continue from `batch_start` through `batch_end`.
- After the worker exits, validate the new rounds created for that batch.
- If `completed_rounds` still remains below `min_rounds`, launch another batch.

The intended steady-state behavior is one worker launch per batch. If a worker stalls, exits early, or must repair an invalid round, the CLI may relaunch a worker to finish or repair the same logical batch.

### Accepted Progress Uses Validated Round Prefixes

Batch progress should not be measured by counting all newly created `opt-round-*` directories.

- The CLI must inspect new rounds in round-number order.
- Progress advances only through the longest valid prefix of the newly created rounds.
- If `opt-round-4` is invalid, then later rounds such as `opt-round-5` do not count toward session progress even if their directories exist.
- The continuation summary should identify the first unresolved round and explain that later rounds in the same batch are not yet accepted as session progress.

This preserves a strict round-by-round contract even when one worker launch creates multiple rounds.

## Worker Prompt Contract

The old one-round worker prompt is not enough for batched execution. The new worker prompt should say:

- this invocation owns rounds `current_round` through `final_round`
- execute those rounds strictly one at a time
- do not pre-plan the full batch before acting
- before each round, re-evaluate the next bottleneck and choose the right analysis depth from the existing evidence ladder
- do not decide session-level stop policy on your own
- the CLI will validate the completed batch after the invocation exits

The prompt should explicitly discourage "batch planning" language such as deciding all remaining round hypotheses up front. The worker may keep short local notes, but each round must still be analyzed and justified independently.

## Checked Batch Flow

After a worker batch exits successfully, the CLI should:

1. Identify which round directories were newly created or modified for the batch.
2. Validate each expected round in order using the round checker.
3. Build a structured batch follow-up summary that includes:
   - accepted rounds in this batch
   - the first invalid or missing round, if any
   - warning-only issues from accepted rounds
   - whether the global `min-rounds` target is satisfied
   - the next round that should be opened
4. Decide the next action:
   - stop successfully if the global target is satisfied and no repair is needed
   - relaunch a worker with a continuation prompt if more rounds are required
   - relaunch a worker with a repair-focused continuation prompt if some round in the batch was invalid
   - fail immediately on hard-fail round results

The continuation summary should align with the current checked-mode pattern: the CLI provides direct, concrete repair instructions instead of silently accepting invalid state.

## Supervised Batch Flow

`supervised` keeps the same worker batch flow as `checked`, then adds one batch-level supervisor pass.

### Supervisor Timing

- Launch the supervisor once after the CLI has validated the batch's rounds.
- Pass the CLI batch follow-up summary into the supervisor prompt.
- The supervisor reads the new batch rounds, not only the latest round.

### Supervisor Scope

The supervisor should audit the batch as a unit:

- whether the accepted rounds respect the workflow contract
- whether the first invalid round, if any, is accurately diagnosed
- whether warnings from accepted rounds need metadata-only correction or explicit carry-forward guidance
- whether the next worker batch should repair, continue, or stop

The supervisor still must not perform open-ended optimization or invent missing evidence.

### Supervisor Outputs

The supervisor should keep using `.helix/supervisor-report.md`, but the prompt and runtime should treat it as the audit report for the last completed batch rather than only for the last completed round.

`--round-batch-size 1` naturally degenerates to the existing per-round supervision semantics because each batch contains exactly one round.

## Round Checker Contract

The round checker should stop accepting `--min-rounds` and instead accept:

- `--current-round`
- `--final-round`
- `--optimize-target`

### Why This Changes

- `--min-rounds` is a session-level policy owned by the CLI.
- `check-round` should validate the current round and provide local forward guidance, not decide whether the whole optimize session may stop based on workspace-wide round counts.
- Passing explicit round numbers makes batch prompts and checker summaries deterministic even if invalid or ignored round directories already exist in the workspace.

### New Checker Behavior

When `current_round` and `final_round` are provided:

- the checker validates the current round exactly as before for artifacts, semantic fields, continuity, and warning-style issues
- on pass, it emits summary text relative to `current_round` and `final_round`
- if `current_round < final_round`, the summary should say another round in the current batch is still required and name the next round
- if `current_round == final_round`, the summary should say this was the final required round for the current worker batch

The checker should not decide whether the entire optimize session may stop. That decision moves to the CLI after it merges checker results with total accepted progress.

## Runtime Architecture Changes

### Options And Request Models

- Remove `continuous` from optimize round-mode literals in the CLI-facing option and request models.
- Add `round_batch_size` to optimize option and request models.
- Validate `round_batch_size >= 1`.

### Prompt Builders

- Replace the old continuous-versus-single-round split with batched worker prompt builders.
- The worker prompt builder should accept:
  - `current_round`
  - `final_round`
  - `round_batch_size`
  - `resume_existing_session`
- The supervisor prompt builder should accept:
  - the list or range of batch rounds to audit
  - the CLI batch follow-up summary

### Execution Paths

- Remove optimize's command-surface dependency on the old continuous single-session path.
- Route both `checked` and `supervised` through one shared batched controller.
- Keep supervised-only session artifacts such as `supervisor-report.md` and supervisor history directories.
- Reuse the existing checked-mode continuation-prompt pattern instead of introducing a separate repair agent role.

### Repair Attempt Tracking

The existing repair loop cap should remain, but the tracking key should be the first unresolved round in the current batch rather than only the latest round name. This avoids infinite retries when a batch repeatedly fails on the same round while later stale round directories remain present.

## Testing Strategy

Add or update tests for:

- CLI parsing and defaults
  - `--round-mode` accepts only `checked` and `supervised`
  - default round mode is `checked`
  - `--round-batch-size` defaults to `10`
  - invalid batch size is rejected
  - `--interact` is rejected for optimize
- prompt generation
  - worker prompts mention `current_round` and `final_round`
  - worker prompts say not to pre-plan the whole batch
  - supervisor prompts describe batch-level auditing
- round checker
  - `check-round` uses `current_round` and `final_round`
  - summaries mention the next round inside the current batch when appropriate
  - the checker no longer emits session-stop decisions from `min_rounds`
- runtime orchestration
  - checked mode validates every round in a batch
  - invalid intermediate rounds block acceptance of later rounds in the same batch
  - supervised mode launches one supervisor per batch
  - `round_batch_size=1` degenerates to one worker batch plus one supervisor batch audit per round
  - the next worker prompt carries CLI warnings and supervisor guidance into the next batch
- batch optimize plumbing
  - `optimize-batch` preserves `round_mode` and `round_batch_size`

## Migration And Compatibility

- Existing invocations that explicitly pass `--round-mode continuous` will fail with an argparse choice error after the change.
- Existing default optimize invocations will shift from `continuous` to `checked`.
- The release notes and README optimize sections should explain the new default and the meaning of `--round-batch-size`.
- Historical design docs that mention `continuous` remain as history; they should not be treated as the current command contract.
