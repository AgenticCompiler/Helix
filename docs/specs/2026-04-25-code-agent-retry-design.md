# Code Agent Retry Design

## Goal

Introduce one shared retry mechanism for CLI-launched code agents so transient rate-limit failures are handled consistently across backends, while allowing users to control the retry count with an environment variable.

## Scope

- Include CLI-backed code agent backends that inherit from `AgentRunner`:
  - `codex`
  - `opencode`
  - `pi`
  - `claude`
  - `traecli`
- Exclude `openhands`, which uses a separate in-process execution path.
- Apply only to code agent launches, not to local run-eval helper scripts, test execution, benchmark execution, or other subprocesses outside the backend runner layer.

## Current Problem

Today, transient `429` / rate-limit handling is partially implemented inside the optimize loop. That creates three problems:

1. The behavior is not shared by non-optimize commands such as `convert`, `gen-test`, `gen-bench`, and `gen-eval`.
2. The retry policy is attached to optimize orchestration instead of the code agent launch boundary where the failure actually occurs.
3. Worker and supervisor launches can end up with optimize-specific retry logic that would become duplicated if a shared retry layer is added later.

## Design Summary

Move transient code agent retry handling into `AgentRunner`, so every CLI backend gets the same retry behavior for non-interactive launches. Keep optimize orchestration focused on orchestration-only recovery such as stalled runs and minimum-round continuation.

## Retry Trigger

The shared retry layer should treat a non-interactive agent run as transiently retryable when all of the following are true:

- The request is non-interactive.
- The result is not marked `stalled`.
- The result return code is not `130`.
- The combined stdout/stderr text contains one of these case-insensitive patterns:
  - `429 too many requests`
  - `exceeded retry limit`
  - `rate limit`

The initial implementation should keep this pattern list aligned with the current optimize behavior. If later backends need broader transient detection, that should be added centrally in the same shared retry helper rather than reintroduced in command-specific orchestration.

## Retry Count Control

Add one environment variable:

- `TRITON_AGENT_CODE_AGENT_MAX_RETRIES`

Semantics:

- Unset: default to `2`.
- `0`: disable automatic retry.
- Positive integer `N`: allow up to `N` extra attempts after the initial launch.
- Negative integers and non-integer values: raise a clear `ValueError` with the variable name and invalid value.

This preserves the current effective optimize default while making the policy visible and adjustable for all CLI code agent launches.

## Backoff Policy

Use the same exponential backoff currently used by optimize:

- retry 1 waits `1` second
- retry 2 waits `2` seconds
- retry 3 waits `4` seconds

Formula: `2 ** (retry_number - 1)`.

This should live beside the shared retry helper so the policy is defined in one place.

## Layering

### Backend layer

`AgentRunner.run()` should become the shared entrypoint for retryable CLI launches:

- Build the command.
- Log the launch command once when verbose mode is enabled.
- Execute the child process.
- If the result is transient and retries remain, sleep and launch again.
- Return the final `AgentResult`.

Interactive mode should bypass retry and preserve current behavior.

The retry loop should wrap both `run()` and `resume()` automatically because `resume()` already delegates back into `run()` with a rewritten prompt.

### Optimize layer

`OptimizeRunLoop` should stop treating rate-limit text as its own recovery category.

After this change, optimize should only keep orchestration-owned recovery:

- stalled worker recovery
- stalled unsupervised recovery
- minimum-round continuation
- supervisor decision handling (`PASS_STOP`, `PASS_CONTINUE`, `REVISE_REQUIRED`, `REVISE_METADATA`)

Supervisor agent launches should use the shared backend retry path automatically, so optimize should not perform a second 429-specific retry after converting supervisor failures into `GateResult`.

## Failure Semantics

- If retries are exhausted, return the last `AgentResult` unchanged.
- The CLI command should continue to print stdout/stderr and return the final backend exit code exactly as it does today.
- Retry should not rewrite the captured stderr message.
- Session logging in optimize should continue to record each agent launch attempt, because each retry is a real agent invocation.

## Files Expected To Change

- `src/triton_agent/backends/base.py`
  - add shared retry logic for CLI-backed code agent launches
  - add env-var parsing for retry count or delegate it to a small local helper
- `src/triton_agent/optimize/run_loop.py`
  - remove 429 / rate-limit detection and backoff handling
  - keep stall and orchestration-only continuation logic
- `tests/test_backends_base.py`
  - add coverage for shared retry behavior
- `tests/test_supervisor.py`
  - update optimize-loop tests so they no longer expect optimize-owned 429 retry behavior

## Testing Strategy

Add or update tests for these cases:

1. Non-interactive backend run retries transient rate-limit failures and eventually succeeds.
2. Retry count `0` disables automatic retry.
3. Retry delay uses the shared exponential sequence.
4. Interactive runs do not retry.
5. Invalid `TRITON_AGENT_CODE_AGENT_MAX_RETRIES` values fail clearly.
6. Optimize unsupervised flow still retries stalled runs via resume.
7. Optimize supervised flow still handles supervisor gate decisions, but no longer owns a second rate-limit retry layer.

## Non-Goals

- Adding jitter, max-delay caps, or backend-specific retry policies.
- Retrying `openhands`.
- Retrying local run-eval subprocesses.
- Expanding transient detection beyond the current rate-limit-oriented patterns.

## Risks And Mitigations

### Risk: double retry

If optimize keeps its old 429 logic after the backend retry layer lands, worker and supervisor launches may retry twice.

Mitigation: remove rate-limit retry detection from `OptimizeRunLoop` in the same change.

### Risk: accidental widening of retry scope

If retry is implemented in `process_runner`, non-agent subprocesses could start retrying unexpectedly.

Mitigation: keep retry in `AgentRunner`, not in the generic process runner.

### Risk: hidden configuration errors

If invalid env-var values silently fall back to defaults, users may believe retries are configured when they are not.

Mitigation: fail explicitly on invalid values.

## Open Questions Resolved

- Include `openhands`? No.
- Should supervisor launches use the same retry path? Yes.
- Should optimize still own 429 retry after this change? No.
