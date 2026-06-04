# Remote Execution Env Fallback

## Summary

When an outer agent-driven command is launched with `--remote`, the CLI should also inject the remote context into the agent subprocess environment. Remote-aware execution helpers must then treat that environment as a fallback when the agent forgets to pass `--remote` or `--remote-workdir` explicitly.

## User-Visible Behavior

- Agent-driven commands that already accept `--remote` and `--remote-workdir` keep the same prompt guidance.
- In addition, they export:
  - `TRITON_AGENT_REMOTE`
  - `TRITON_AGENT_REMOTE_WORKDIR`
- Remote-aware helper commands resolve execution context with this precedence:
  1. explicit CLI flags
  2. injected environment variables
  3. local execution
- If no remote target is resolved, `remote_workdir` is ignored.

## Scope

- Apply the fallback to staged `triton-npu-run-eval` helper commands such as `run-test-*`, `run-bench`, `profile-bench`, and `compare-result`.
- Apply the same fallback to top-level CLI wrappers that may be called by agents instead of staged helper scripts.
- Do not change SSH/SCP implementation details or the actual remote execution flow.

## Implementation Notes

- Keep the env-variable names and resolution rules in one loadable helper under `skills/triton-npu-run-eval/scripts/`.
- Reuse that helper from `src/` through the existing skill loader bridge so the contract is not duplicated.
- Explicit flags must continue to override the environment fallback.
