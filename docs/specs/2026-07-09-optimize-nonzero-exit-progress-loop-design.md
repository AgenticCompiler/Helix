# Optimize Nonzero Exit Progress Loop Design

## Summary

- Simplify checked and supervised optimize round-loop control so worker agent exit status no longer directly decides whether the session stops.
- After every worker agent exit, re-scan the workspace and decide whether to continue only from session progress: attempted round count, best accepted-round geomean speedup, and no-progress retry budget.
- Keep network failures, stalls, and generic non-zero exits as diagnostic facts recorded in logs, not as separate orchestration branches.
- Define explicit enum ranges for `correctness_status` and `benchmark_status` so the worker must admit whether a round passed, failed, or never reached that check.
- Extend the existing per-invocation `trace-*.jsonl` logs with optimize controller events so launch, progress checks, continue decisions, and stop reasons are directly visible during debugging.
- Remove now-unused round-loop recovery concepts and tests once the simplified controller flow is in place.

## Goals

- Make optimize continuation behavior easy to explain: after each agent exit, inspect current valid progress and continue until stop conditions are met.
- Allow sessions to continue after worker `return_code != 0` when the workspace has still made valid optimize progress.
- Count explicit terminal rejected rounds toward `--min-rounds` so the session does not get stuck retrying the same abandoned attempt forever.
- Preserve `--min-speedup` as an early-success target that may stop the session before `--min-rounds`.
- Prevent infinite loops by stopping after a bounded number of consecutive worker exits with no new valid progress.
- Make optimize state contracts explicit enough that the worker can honestly close a failed round instead of leaving free-form or implicit state behind.
- Improve observability of optimize orchestration without adding wrapper noise to `show-output` logs.

## Non-Goals

- Do not change `continuous` or `--interact` optimize execution semantics in this change.
- Do not change accepted-round speedup semantics; keep `passed` plus `passed` as the only speedup-bearing round outcome.
- Do not add a new standalone optimize loop log file.
- Do not redesign backend-level retry behavior for non-optimize commands.
- Do not introduce a second failure taxonomy such as `fatal`, `retryable`, `abandoned`, or `stall` into round-state contracts.

## Problem

The current checked/supervised round loop mixes two different control systems:

- the worker exit result
- the session-level optimize progress

Today, `_run_worker_with_recovery(...)` uses failure labels such as `stall`, `transient`, and `fatal` to decide whether optimize should retry the same range or stop immediately. That creates two problems:

1. optimize may stop early even when the workspace already contains enough valid progress to continue or even succeed
2. debugging is difficult because the true stop condition is distributed across worker exit classification, recovery budgets, batch checks, and prompt injection

There is also a second correctness problem in the current round-state model:

- a round only counts when `correctness_status == "passed"` and `benchmark_status == "passed"`
- workflow state only lets a round transition from `active` to `passed`

That means a worker can honestly try a round, discover correctness failure, and give up, but the controller still sees "no valid round happened" and may keep retrying the same slot.

The desired behavior is simpler: after the worker exits, inspect the workspace and decide from actual optimize progress whether to continue, succeed, or fail.

## User-Visible Behavior

### Checked And Supervised Round Loops

For `checked` and `supervised` optimize runs:

1. launch the worker agent for the current scheduled range
2. wait for the worker agent to exit
3. inspect the workspace for current valid optimize progress
4. decide whether to continue or stop from that session state

Worker exit status remains visible in logs, but it no longer directly ends the optimize session.

### Stop Conditions

After each worker exit, the controller evaluates the following conditions in order:

1. if `--min-speedup` is configured and the current best completed-round geomean speedup is at least that target, stop successfully
2. else if the current attempted round count is at least `--min-rounds`, stop successfully
3. else if the worker produced no new terminal round progress and the consecutive no-progress attempt count has reached the configured limit, stop with failure
4. else continue with the next worker invocation

### Worker Exit Examples

- worker exits with code `1`, but one new rejected terminal round appears: continue
- worker exits with code `0`, but no new terminal round appears: count as no progress
- worker exits with code `1`, no new terminal round appears, and the no-progress budget is exhausted: fail
- worker exits with code `1`, but the current best speedup already satisfies `--min-speedup`: succeed

## Progress Model

### Attempted And Accepted Rounds

The runtime needs two different counts:

- `attempted_round_count`: rounds that reached an explicit terminal outcome, even if they were rejected
- `accepted_round_count`: rounds that satisfy the current strict accepted-round semantics and may contribute speedup

Implementation direction:

- derive `attempted_round_count` from a new terminal-round helper, not from raw `opt-round-*` directory counts
- keep `best_completed_round_geomean_speedup(...)` as the authority for accepted-round speedup-based stopping
- treat the current completed-round helper path as the source of `accepted_round_count`

### Progress Made

For this change, the progress predicate stays intentionally simple:

- compute `attempted_round_count_before` immediately before launching the worker
- compute `attempted_round_count_after` immediately after the worker exits
- `progress_made = attempted_round_count_after > attempted_round_count_before`

Accepted-round growth is still useful for speedup checks and logs, but it is not required for `progress_made`.

### Explicit Status Contract

Both `correctness_status` and `benchmark_status` must become explicit enums in the round and baseline state contracts.

Allowed values for both fields:

- `passed`
- `failed`
- `not_run`

The round and baseline contract JSON files should expose these ranges as machine-readable enum arrays, not only as prose descriptions, so prompt builders, checkers, and runtime code share one source of truth.

Semantics:

- `passed`: that check ran and passed
- `failed`: that check ran and failed
- `not_run`: that check never reached a conclusive run in this round or baseline

The worker must always write one of these values explicitly. Missing fields, free-form strings, or any value outside this enum range are contract violations and do not count as progress.

For baseline state, `baseline_established` remains `true` only when both statuses are `passed`.

There is intentionally no separate `abandoned` enum. If the worker gives up on a round, it should admit that outcome with `failed` or `not_run` and explain the reason in the round summary and attempt notes.

### Terminal Versus Accepted Rounds

Round outcomes are derived from the status pair, not from worker exit code:

- accepted round: `correctness_status == "passed"` and `benchmark_status == "passed"`
- rejected terminal round: any other enum-valid status pair
- unresolved round: missing round state, missing minimum artifacts, invalid JSON, or out-of-range status values

Controller consequences:

- accepted rounds count toward `attempted_round_count`
- rejected terminal rounds also count toward `attempted_round_count`
- only accepted rounds count toward `accepted_round_count` and speedup evaluation
- unresolved rounds do not count toward progress

The accepted-round checker should stay strict. Rejected terminal rounds are not "passing" rounds; they are only session progress.

### No-Progress Attempt Budget

Add a controller-owned limit for consecutive worker exits that produce no new valid progress.

Behavior:

- when `progress_made` is `true`, reset `no_progress_attempts` to `0`
- when `progress_made` is `false`, increment `no_progress_attempts`
- if `no_progress_attempts` reaches the configured limit, stop with failure

The default should be a small fixed value such as `3`, owned inside optimize runtime code for now.

## Runtime Design

### Controller Flow

Replace the current worker recovery branch structure with a single post-exit session-progress flow:

1. record pre-launch session progress
2. run the worker once for the scheduled batch
3. record raw worker result facts
4. recompute post-exit session progress
5. update the no-progress budget
6. decide continue / succeed / fail from session state

This means the round-loop controller no longer needs to classify optimize worker exits into orchestration labels such as `fatal`, `stall`, or `transient`.

### Batch Scheduling

Keep the existing `round_batch_size` and `--min-speedup` policy, but recompute the next range from session state after each worker exit:

- preserve single-round dispatch when `--min-speedup` is active
- preserve checked versus supervised differences around supervisor audit
- derive `batch_start` from `attempted_round_count + 1`, not from accepted-round count
- derive `batch_end` from the existing `round_batch_size` rules using that recomputed `batch_start`
- never automatically reopen a rejected terminal round just because it was not accepted

This is necessary so a round that was honestly rejected still advances the session to the next round number.

### Batch Check Position

The technical accepted-round check remains useful, but its role becomes narrower:

- keep it as the authority for deciding whether a round is accepted
- do not overload it with the separate question "did the worker produce a terminal round outcome?"
- allow the controller to continue later iterations as long as stop conditions are not yet met and the no-progress budget is not exhausted

Implementation direction:

- keep `check_round(...)` strict so it still fails rejected rounds
- add a separate terminal-round inspection helper for controller progress accounting and workflow-state closure

### Workflow-State Closure

Current workflow state only allows `active` and `passed` round statuses. That is too strict for the new continuation model because a rejected terminal round still needs to close cleanly so the next round can start.

Implementation direction:

- extend workflow-state round status to allow `failed` as a terminal non-accepted outcome
- close the active round as `passed` when `check_round(...)` passes
- close the active round as `failed` when the round is terminal but rejected
- keep the optimize session blocked only when the round is unresolved, not when it is explicitly rejected

This keeps workflow state simple while still allowing the controller to move forward after an honest failed attempt.

### State-Check Failure

If the controller cannot compute post-exit session state at all, that is not treated as "no progress". It is an orchestration failure.

Examples:

- completed-round helper raises unexpectedly
- best-speedup helper raises unexpectedly
- required optimize session state cannot be read

In those cases, stop immediately with an explicit orchestration error and log the failure reason.

## Logging Design

### Reuse Existing Trace Files

Do not add a separate optimize-loop log file. Reuse the existing per-launch `trace-*.jsonl` files already archived under `triton-agent-logs/<run-id>/`.

Append optimize controller events to the same trace stream so one file shows both:

- backend/agent invocation evidence
- controller orchestration evidence

### New Controller Events

Add compact structured trace events such as:

`optimize_loop_start`

- `min_rounds`
- `min_speedup`
- `max_no_progress_attempts`
- `round_mode`

`optimize_iteration_start`

- `iteration`
- `batch_start`
- `batch_end`
- `attempted_round_count_before`
- `accepted_round_count_before`
- `best_speedup_before`
- `no_progress_attempts_before`

`optimize_worker_result`

- `iteration`
- `batch_start`
- `batch_end`
- `return_code`
- `stalled`
- `retryable_failure`
- `session_id`

`optimize_progress_check`

- `iteration`
- `attempted_round_count_before`
- `attempted_round_count_after`
- `accepted_round_count_before`
- `accepted_round_count_after`
- `best_speedup_after`
- `latest_round_outcome`
- `progress_made`

`optimize_iteration_decision`

- `iteration`
- `decision`
- `reason`
- `no_progress_attempts_after`

Decision values:

- `continue`
- `stop_success_min_speedup`
- `stop_success_min_rounds`
- `stop_failed_no_progress_limit`
- `stop_failed_state_check`

`optimize_loop_stop`

- `final_attempted_round_count`
- `final_accepted_round_count`
- `final_best_speedup`
- `final_return_code`
- `decision`

### Logging Scope

Network errors, stalls, and non-zero exits should still be visible, but only as raw worker-result fields in logs. They are no longer control-flow concepts in optimize runtime code.

## Code Cleanup Plan

This change should remove dead or misleading round-loop recovery code once replacement tests pass.

Expected cleanup targets include:

- optimize-specific use of `classify_worker_failure(...)`
- optimize-specific use of `RecoveryBudget`
- optimize-specific transient/stall recovery prompt note builders
- optimize-specific branches that stop early from `fatal`
- any optimize-only helpers that assume "latest round" always means "latest accepted round"

Keep backend-level retry infrastructure for non-optimize commands intact unless a specific optimize-only path no longer needs to thread that state.

Tests and helper code that only exist to support the removed failure-label controller flow should be deleted or rewritten to match the new progress-based loop.

## Testing Strategy

Add focused runtime tests that verify the new controller semantics:

1. worker exits non-zero and creates one new valid completed round: continue
2. worker exits non-zero and creates one new rejected terminal round: continue and reset the no-progress counter
3. worker exits zero and creates no new terminal round: increment no-progress attempts
4. repeated no-progress exits reach the limit: fail with `stop_failed_no_progress_limit`
5. worker exits non-zero but accepted-round speedup already satisfies `min_speedup`: succeed early
6. worker exits non-zero and attempted round count reaches `min_rounds`: succeed
7. a later progress-making iteration resets the no-progress counter
8. controller session-state recomputation failure stops immediately with explicit error
9. terminal rejected rounds advance batch scheduling so the next invocation starts from the next round number
10. `submit-round` closes workflow state for rejected terminal rounds while still returning non-zero
11. enum violations such as `correctness_status="maybe"` do not count as progress
12. trace output contains the new optimize controller events with expected decision fields

Retain existing tests only when they still describe valid behavior. Delete tests that assert the old stall/transient/fatal orchestration policy.

## Risks And Tradeoffs

- A worker that repeatedly exits with partial or invalid artifacts will now continue until the no-progress budget is exhausted instead of stopping immediately from failure classification.
- Counting rejected terminal rounds toward `min_rounds` means a session may finish without creating many accepted rounds. That is intentional because `min_rounds` is an iteration budget, not a success-quality guarantee.
- Reusing the existing trace file avoids file sprawl, but trace-summary helpers may need small updates so the new controller events are summarized clearly.

## Recommendation

Implement the smallest possible controller simplification:

- progress-based continue/stop decisions
- explicit `passed` / `failed` / `not_run` status enums
- attempted-round versus accepted-round accounting
- workflow-state closure for rejected terminal rounds
- no-progress budget
- controller trace events
- deletion of obsolete optimize-only failure-classification code

This keeps the optimize loop easy to reason about and aligns directly with the intended rule: after each agent exit, inspect current valid progress and continue until success targets are met or progress has clearly stalled out.
