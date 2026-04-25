# Optimize Graceful Interrupt

## Summary

- Add graceful `Ctrl+C` handling for `optimize` code-agent runs.
- A single user `Ctrl+C` should trigger an internal shutdown sequence for the code agent instead of immediately aborting the CLI process.
- The shutdown sequence should send two `SIGINT` signals to the code agent, then force-kill it if it still does not exit.

## User-Visible Behavior

- During `uv run triton-agent optimize ...`, pressing `Ctrl+C` once should ask the CLI to stop the running code agent gracefully.
- The CLI should send one `SIGINT` to the code agent, wait briefly, then send a second `SIGINT` automatically if the agent is still running.
- If the code agent still has not exited after the second grace period, the CLI should force termination with `SIGKILL`.
- The interrupted optimize run should exit as a user interrupt rather than a stall, and the optimize supervisor must not restart the agent for recovery after this path.
- Non-optimize commands should keep their existing interrupt behavior unless they opt into the same lifecycle explicitly in the future.

## Implementation Notes

- Keep the subprocess lifecycle in `src/triton_agent/process_runner.py`, because that module already owns `Popen`, PTY handling, stall detection, and shutdown.
- Add an opt-in interrupt policy for non-interactive process execution so optimize can request the graceful interrupt sequence without changing other commands.
- Catch `KeyboardInterrupt` around the process wait loop, send the configured signals to the child, wait between signals, and return an `AgentResult` that represents a user interrupt.
- Use process groups for non-interactive child processes so the signal sequence reaches the full code-agent subprocess tree instead of only the top-level launcher.
- Keep interactive mode unchanged for now; this change only targets optimize code-agent orchestration.

## Verification

- Process-runner tests for streaming and buffered interruption paths.
- Supervisor test proving user interrupts do not enter stall recovery.
- Optimize runtime or command test proving optimize requests opt into the graceful interrupt policy.
- Update `README.md` and `AGENTS.md` to document optimize interrupt behavior.
