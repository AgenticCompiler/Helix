# Optimize Stall Recovery And Progress Detection Design

## Summary

- Keep shared backend retry focused on generic non-interactive code-agent failures by default.
- Add an optimize-owned worker recovery mechanism that handles both transient backend failures and stalled worker invocations in one place.
- When an optimize worker stalls, recover from the first unresolved round in the current target range instead of failing the whole session immediately.
- Improve stall detection for optimize worker invocations by treating selected optimize artifact file activity as forward progress in addition to agent output.

## Goals

- Let long optimize sessions continue after a worker invocation stalls instead of failing the whole run immediately.
- Preserve already accepted optimize rounds and continue from the first unresolved round.
- Use one optimize-local recovery model for `429` / rate-limit failures and `stall` failures so budgets, prompts, and logs are easier to explain.
- Reduce false-positive stall detection when the agent is still writing meaningful optimize artifacts without producing terminal output.
- Keep the CLI thin by limiting optimize-specific recovery semantics to the optimize feature package rather than widening generic backend behavior.

## Non-Goals

- Do not make stalled retries a shared behavior for non-optimize commands such as `gen-test`, `convert`, or `gen-eval`.
- Do not redefine optimize artifact contracts such as `baseline/`, `opt-round-*`, `round-state.json`, `summary.md`, or `attempts.md`.
- Do not add a new user-facing CLI flag in this change.
- Do not persist new recovery-state files to disk.
- Do not redesign baseline preflight validation or supervisor audit semantics beyond the minimum wiring needed to avoid conflicts with the new worker recovery path.

## Problem

Current optimize worker launches fail immediately when the underlying process runner reports `stalled=True`.

That is too strict for long-running optimize sessions:

- a worker can make real round progress and then go quiet long enough to trip the stall timeout
- a user may request many rounds, so failing the entire session at round 14 of 30 wastes already accepted work
- optimize already has artifact-driven round structure, so it can recover from disk state more precisely than generic commands can

Current stall detection also only considers agent output activity. That can misclassify a worker as stalled while it is still producing useful optimize artifacts under the workspace.

## User-Visible Semantics

### Scope

This design applies only to optimize round worker invocations in the existing multi-invocation optimize flow.

It does not change generic retry behavior for other commands.

### Recoverable worker failures

Optimize worker invocations should treat these results as recoverable:

- transient backend failures detected from existing rate-limit text patterns:
  - `429 too many requests`
  - `exceeded retry limit`
  - `rate limit`
- worker invocations that return `stalled=True`

These two categories share one optimize-owned recovery mechanism.

Recovery classification precedence should be explicit:

1. if `result.stalled` is true, use stall recovery
2. else if the result matches transient backend failure detection, use transient recovery
3. else treat the failure as non-recoverable

This keeps optimize behavior deterministic even when a stalled result also contains rate-limit text in captured output.

### Recovery point

When a worker invocation needs recovery, the CLI should recompute accepted progress from disk instead of trusting the interrupted invocation's intended batch range.

Recovery point rules:

1. Scan the current worker target round range from its starting round upward.
2. For each round, require the round directory to exist and `check_round(...)` to pass.
3. Stop at the first missing or failing round.
4. Treat the longest passing prefix as accepted progress.
5. Restart from `last_accepted_round + 1`.

Example:

- target session goal: 30 rounds
- current worker target: rounds 11 through 15
- rounds 11, 12, and 13 pass `check_round(...)`
- round 14 exists but does not pass, or the invocation stalled before it was accepted

Recovery should resume from rounds 14 through 15.

Already accepted rounds must not be rerun.

This longest-passing-prefix helper is range-local rather than session-global:

- when the current worker target is rounds 11 through 15, the helper starts scanning at 11 rather than round 1
- rounds accepted before the current worker target range are assumed to remain accepted for this helper

### Recovery prompt

Recovered worker invocations should use a fresh batch prompt plus a short CLI recovery note.

For `stall` recovery, the note should say:

- the previous invocation stalled
- which rounds are already accepted
- which round the new invocation must resume from
- that the worker should inspect existing artifacts for the unresolved round before deciding whether to repair or finish them

For transient backend recovery, the note should say:

- the previous invocation ended in a transient backend failure
- the current target range is being retried

### Recovery budget

Optimize should apply a bounded recovery budget per unresolved round.

Default policy:

- each unresolved round gets at most `3` recovery attempts
- the budget key is the first unresolved round after recomputing the longest passing prefix
- both `stall` and transient backend failures consume this same budget
- once accepted progress advances to the next unresolved round, that next round receives a fresh budget

Example:

- round 14 stalls twice and then hits one `429` retryable failure
- round 14 has consumed all `3` recoveries
- if round 14 is still not accepted after that, the optimize session fails
- if round 14 becomes accepted, round 15 starts with a fresh budget of `3`

### File-activity-based progress

Optimize worker stall detection should consider both:

- normal process output activity
- selected optimize artifact file activity

This means a worker should not be classified as stalled when it is still producing meaningful optimize artifacts even if terminal output is quiet.

## Design

### Layering

Keep the responsibility split explicit:

- `process_runner.py` owns generic stall timing and subprocess liveness
- `backends/base.py` owns default shared backend retry for ordinary commands
- `optimize/` owns optimize-specific recovery semantics, artifact progress rules, and round-level continuation

This avoids pushing optimize-only recovery meaning into the generic backend layer.

### Shared backend retry boundary

Shared backend retry should remain the default behavior for non-optimize requests.

Optimize round worker requests should disable shared backend retry so optimize can own one combined recovery loop for:

- transient backend failures
- stalled worker runs

Baseline repair launches and supervisor launches should keep current behavior in the first version:

- they may continue to use shared backend transient retry
- they do not participate in round-prefix recovery semantics

### Optimize worker recovery loop

The optimize controller should wrap worker invocations in a recovery loop.

High-level flow:

1. Build the worker request for the current round range.
2. Run the worker once.
3. If it succeeds, continue with normal batch validation.
4. If it fails with a recoverable optimize worker failure:
   - recompute accepted progress when needed
   - consume recovery budget for the first unresolved round
   - rebuild the worker request with recovery guidance
   - rerun from the adjusted range
5. If it fails with a non-recoverable result or the per-round budget is exhausted, fail the optimize session.

### Longest passing prefix helper

Add one optimize-local helper that computes accepted progress for a requested round range.

The helper should:

- scan rounds in numeric order from `batch_start` through `batch_end`
- stop at the first missing round directory
- stop at the first round whose `check_round(...)` result is not `pass`
- return:
  - `last_accepted_round`
  - `first_unresolved_round`
  - adjusted recovery range bounds

This helper should use the same round validation authority that normal optimize batch checks already use.

### File activity probe

`process_runner.py` should accept an optional generic progress probe callback.

The callback contract should stay generic, for example:

- return the most recent externally observed progress timestamp
- or return `None` when no additional progress source is configured

The stall timer should reset when either:

- process output activity happens
- the external progress probe reports newer activity than the last observed value

`process_runner.py` must not contain optimize-specific file-name knowledge.

The progress probe should run inline inside the existing stall polling loops for buffered and streaming execution paths.

This version should not introduce a separate probe thread. The polling loop already sleeps briefly between checks, and inline probe calls keep synchronization simpler.

### Optimize file activity rules

Optimize should provide a worker-only progress probe that scans a narrow whitelist of business artifacts.

Use concrete allow-list roots under the optimize workspace:

- `baseline/**`
- `opt-round-*/**`
- `opt-note.md`
- `learned_lessons.md`

Count as progress:

- files under `baseline/`
- files under `opt-round-*`
- `opt-note.md`
- `learned_lessons.md`
- round-local business artifacts such as:
  - `round-state.json`
  - `summary.md`
  - `attempts.md`
  - perf outputs
  - profiler outputs
  - IR outputs
  - analysis artifacts

Do not count as progress:

- `helix-logs/`
- `.helix/`
- show-output logs
- trace files
- session metadata files
- temporary cleanup noise that is not a meaningful optimize artifact update

The probe should treat file progress conservatively:

- count file creation
- count file size change
- count forward-moving file `mtime`
- do not treat directory `mtime` alone as progress

Implementation should prefer these explicit allow-list roots over loose extension-based heuristics.

### Request wiring

Add small request-level plumbing so optimize worker launches can:

- disable shared backend retry
- provide optimize progress-source configuration to the subprocess layer

The request model should stay explicit rather than encoding this through command-name conditionals inside the backend base layer.

Implementation should not store a live Python callback on `AgentRequest`.

Preferred shape:

- keep a request-level boolean opt-out for shared backend retry
- keep a request-level progress-source configuration object
- construct the actual progress probe callable in the runner / process layer from that configuration

### Prompt construction

Add optimize-local helpers for short recovery notes:

- transient backend retry note
- stall recovery note

These notes should append to the fresh worker batch prompt rather than reuse the old resume prompt path.

The goal is to keep recovered invocations aligned with the current multi-invocation optimize structure.

### Logging and trace expectations

Each recovery attempt remains a real agent launch.

Therefore:

- existing per-launch show-output logs should continue to record each launch separately
- existing per-launch trace files should continue to record each launch separately
- recovery attempts should remain visible in optimize session artifacts

No special hidden retry path should bypass normal launch recording.

## Module Changes

### `src/helix/process_runner.py`

- Add an optional generic external progress probe input to buffered and streaming process execution.
- Refresh stall timing from probe-reported activity as well as process output.
- Keep default behavior unchanged when no probe is provided.

### `src/helix/backends/base.py`

- Preserve shared transient retry as the default for ordinary requests.
- Respect a request-level opt-out so optimize worker launches can bypass shared backend retry.

### `src/helix/models.py`

- Add explicit request fields for:
  - disabling shared backend retry
  - passing process progress-source configuration

Preferred field shape:

- `disable_backend_retry: bool = False`
- `progress_source: ProgressSourceConfig | None = None`

The exact helper type name may vary, but the model should carry configuration rather than a live callback.

### `src/helix/optimize/execution.py`

- Wrap worker launches in an optimize-owned recoverable failure loop.
- Use the new longest-passing-prefix helper to recompute accepted progress after stalled runs.
- Track per-unresolved-round recovery budgets.
- Rebuild worker request bounds and recovery prompt notes after recoverable failures.

### `src/helix/optimize/recovery.py`

Add a new optimize-local module to keep `execution.py` focused.

Suggested responsibilities:

- recoverable failure classification for optimize workers
- per-round recovery budget tracking
- longest passing prefix computation
- optimize progress probe implementation and whitelist rules
- recovery prompt note rendering

This follows the repository preference for feature-local ownership instead of expanding generic top-level helpers.

## State Tracking

- Do not write a new recovery state file.
- Keep recovery counters in process memory for the lifetime of one CLI optimize run.
- Recompute accepted round progress from disk whenever recovery needs to decide where to resume.

This keeps restart behavior simple:

- if the CLI process exits completely, a later `--resume continue` run still rebuilds state from real artifacts
- recovery does not depend on hidden in-memory-only acceptance records surviving across separate CLI invocations
- recovery budget resets on a fresh CLI restart by design, so an unresolved round can receive a new local recovery budget after the optimize command is restarted

This asymmetry is intentional: accepted artifact progress survives process restarts, but in-memory retry exhaustion does not.

## Testing Strategy

Add or update tests for these cases.

### Process runner

- output is quiet but optimize business files keep changing, so the process is not marked stalled
- only log or trace files change, so the process can still be marked stalled
- no output and no business file progress still triggers stall
- the external progress probe runs inline inside the existing stall polling loops rather than from a separate thread

### Backend base

- shared transient retry still works for ordinary non-optimize requests
- optimize worker requests can disable shared backend retry

### Optimize execution

- worker transient failures are recovered by the optimize controller instead of being consumed by backend shared retry
- worker stalls recover from the first unresolved round in the current target range
- already accepted rounds are not rerun
- recovery prompt text includes the appropriate transient or stall guidance
- recovery budget is tracked per unresolved round and resets when accepted progress advances
- budget exhaustion for one round terminates the optimize session clearly
- baseline repair and supervisor launches do not accidentally enter round-prefix recovery behavior
- business artifact changes such as `round-state.json`, `summary.md`, and `attempts.md` reset stall timing through the optimize progress source
- excluded paths such as `helix-logs/` and `.helix/` do not reset stall timing
- directory `mtime` changes alone do not reset stall timing
- non-recoverable worker-launch failures exit the recovery loop immediately
- worker-launch success followed by ordinary batch validation failure stays on the existing post-run batch failure path rather than re-entering worker recovery

### High-value regression case

Cover this specific scenario:

- requested worker range is rounds 11 through 15
- rounds 11 through 13 are accepted
- round 14 invocation stalls
- the next worker invocation must target rounds 14 through 15

It must not restart at 11 through 15.
It must not skip directly to 15 through 15.

## Risks And Mitigations

### Risk: double retry between backend and optimize

If optimize worker launches still use shared backend transient retry, rate-limit failures may be retried both in the backend and in optimize.

Mitigation:

- add an explicit request-level opt-out from shared backend retry
- enable that opt-out only for optimize worker launches in this change

### Risk: false progress from non-business files

If all workspace file activity counts as progress, logs and trace files may suppress legitimate stall detection.

Mitigation:

- use an optimize-local whitelist of business artifacts
- explicitly exclude `helix-logs/` and `.helix/`

### Risk: skipping an invalid round

If recovery uses directory existence alone, it may continue past an incomplete or invalid round.

Mitigation:

- define accepted progress using the longest continuous prefix that passes `check_round(...)`

### Risk: infinite or excessive recovery loops

If recoverable failures have no bounded budget, one unresolved round may loop forever.

Mitigation:

- enforce a per-unresolved-round budget of `3` recoveries by default

### Risk: growing `execution.py` complexity

If recovery logic is implemented inline, optimize execution code may become harder to understand and test.

Mitigation:

- move recovery-specific helpers into a new optimize-local module

## Verification

Targeted verification should include:

- optimize runtime tests covering worker recovery flow
- backend base tests covering request-level retry opt-out
- process runner tests covering output-plus-file-progress stall behavior

Before claiming implementation complete, run the repository verification commands:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
