## Summary

- During `optimize`, write a temporary workspace `AGENTS.md` that keeps the optimization workflow visible to the code agent throughout a long run.
- If the workspace already has an `AGENTS.md`, back it up before writing the temporary optimize guidance and restore it after the run.

## User-Visible Behavior

- `optimize` injects a short run-specific `AGENTS.md` into the operator workspace before launching the agent.
- The injected guidance should include the actual test mode and benchmark mode chosen for the current optimize run.
- If no workspace `AGENTS.md` exists, the optimize-specific file is removed after the run.
- If a workspace `AGENTS.md` already exists, it is restored after the run and the temporary optimize file is removed.
- Verbose mode should show the backup, temporary write, removal, and restore steps.

## Implementation Notes

- Keep the temporary guidance concise and focused on optimization invariants.
- Limit this behavior to the `optimize` command.
- Use a dedicated manager so preparation and cleanup remain symmetric and testable.
