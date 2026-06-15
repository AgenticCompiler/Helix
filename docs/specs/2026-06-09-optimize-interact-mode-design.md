# Optimize `--interact` Single-Session Behavior

## Summary

`triton-agent optimize --interact` should run as one long attached worker session instead of a batched multi-invocation flow. The CLI must stop rejecting interactive optimize runs and must prepare the optimize request so the agent can stay attached while it completes baseline setup and the requested optimization rounds in one session.

## User-Visible Semantics

- `--interact` is accepted for optimize runs that use the normal checked round flow.
- When `--interact` is set, the CLI forces `round_batch_size` to `99` so the first worker batch spans the remaining requested rounds instead of splitting work across multiple agent launches.
- Interactive optimize skips the standalone baseline preflight-and-repair phase that normally runs before worker batches.
- The worker prompt must explicitly tell the agent to establish or repair `baseline/` inside the same interactive session before `opt-round-1` when needed.
- Report generation remains disabled in interactive mode.

## Implementation Notes

- Keep the change in CLI orchestration rather than moving optimize workflow logic out of skills or prompts.
- Preserve the existing OpenHands restriction for `--interact`.
- Non-interactive optimize behavior remains unchanged.

## Verification

- Command parsing test proving optimize no longer rejects `--interact`.
- Runtime test proving the first interactive batch spans through `min_rounds`.
- Runtime test proving interactive optimize skips baseline preflight and sends the updated worker prompt guidance.
