# Optimize Interact Without Post-Exit Round Checks

## Summary

`helix optimize --interact` should attach the user to one agent session and return that session's exit status without running the CLI round checker after the interactive process exits. The round artifact contract remains unchanged for non-interactive optimize runs.

## User-Visible Semantics

- Interactive optimize still stages the optimize skills and sends the worker prompt.
- Interactive optimize still skips the separate baseline preflight phase so the attached session can establish or repair `baseline/` itself.
- After the interactive process exits, the CLI does not run `check-round`, does not run the supervised audit pass, and does not fail because default later rounds are missing or incomplete.
- The worker prompt and skill contract still require agents to use `submit-round` while completing rounds.
- Non-interactive optimize keeps the existing batch checker behavior.

## Implementation Notes

- Keep the change in `src/helix/optimize/execution.py` by short-circuiting the round loop for interactive requests after the worker invocation returns.
- Do not change optimize prompt semantics for round-internal validation.
- Avoid weakening `ascend-npu-optimize-state` or the round artifact contract; this is only an interactive orchestration behavior change.

## Verification

- Runtime regression test proving an interactive worker can return success without triggering post-exit round checks.
- Prompt regression assertion proving interactive optimize still tells the agent to run `submit-round` for completed rounds.
- Existing non-interactive optimize runtime tests continue to cover checked and supervised batch validation.
