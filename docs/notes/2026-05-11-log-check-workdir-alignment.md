# Log Check Workdir Alignment

## Summary

`log-check` and `log-check-batch` should run each check in the workspace being inspected, not in the repository root.

## Behavior

- The agent backend `workdir` should match the target workspace path.
- The `triton-agent-logs/<command>.show-output.log` file should live under that same workspace.
- Prompt references to repository-owned pattern data should remain explicit and not depend on the current working directory.
