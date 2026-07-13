# Optimize Min Speedup Early Exit Design

## Summary

- Add `--min-speedup X` to `optimize` and `optimize-batch`.
- Treat `--min-speedup` as an early-success target: once the optimize session reaches at least `X` geomean speedup over the baseline, the session may stop immediately even if `--min-rounds` is not yet satisfied.
- Surface that target in the worker prompt, temporary memory file, and `ascend-npu-optimize-state submit-round` guidance so the agent learns when it may stop, without relying on the agent to echo the target value back into `submit-round`.
- Keep the CLI as the final authority: after the worker exits, the CLI re-checks the workspace and exits successfully only when the speedup target is truly satisfied.

## User-Visible Behavior

- `uv run helix optimize ... --min-speedup 1.20` means the session may end as soon as the workspace reaches at least `1.20x` geomean speedup over the baseline.
- When `--min-speedup` is not passed, optimize behavior stays unchanged.
- `--min-speedup` takes precedence over `--min-rounds` for success. If the speedup target is reached early, the optimize command exits successfully even when the minimum round count has not been reached yet.
- When `--min-speedup` is active in checked or supervised round-loop mode, the CLI dispatches one round per worker invocation so the agent can stop immediately after `submit-round` reports that the target is satisfied.
- The optimize runner injects the requested target into worker child processes as `HELIX_OPTIMIZE_MIN_SPEEDUP`, and `submit-round` uses that injected value as the authoritative session target.

## Guidance Contract

- Worker prompts must state the minimum speedup target and say that the session may stop immediately once `submit-round` reports the target is satisfied.
- Worker prompts must tell the agent that the runner injects the target into `submit-round` automatically so the agent should not guess or override the target value.
- The round-loop memory file must repeat that same target so resume and longer sessions do not lose the stopping condition.
- `ascend-npu-optimize-state submit-round` must read `HELIX_OPTIMIZE_MIN_SPEEDUP` when present and include an explicit stop-now guideline when the current session best speedup already satisfies the target.

## Implementation Notes

- Thread `min_speedup: float | None` through:
  - optimize CLI parsing
  - `OptimizeRunOptions`
  - `AgentRequest`
  - request-scoped `extra_env` as `HELIX_OPTIMIZE_MIN_SPEEDUP`
  - prompt builders
  - optimize session artifact guidance rendering
- Add one shared skill-side helper that computes the best completed-round geomean speedup for a workspace using the existing baseline-relative comparison basis recorded in each round.
- Reuse that helper from both:
  - `submit-round` guidance generation
  - CLI round-loop early-exit checks
- Keep the speedup target authority tied to existing baseline-relative geomean speedup semantics; do not introduce a second metric or alternate summary source.

## Verification

- CLI parser and option-mapping tests for `--min-speedup`
- prompt and memory-file tests for the new target wording
- `submit-round` script tests for the stop-now guideline
- optimize round-loop tests for early success before `--min-rounds`
