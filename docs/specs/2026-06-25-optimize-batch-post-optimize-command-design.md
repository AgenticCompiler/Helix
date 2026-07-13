# Optimize Batch Post-Optimize Command Design

## Goal

Add one `optimize-batch` option that accepts a shell command and runs it inside each operator workspace after that workspace's optimize run succeeds.

## User-Facing Behavior

- `helix optimize-batch` accepts `--post-optimize-command "..."`.
- The option is batch-only in this change. Single-workspace `optimize` is unchanged.
- The command runs only when the workspace optimize request returns success.
- The command runs with the operator workspace as its current working directory.
- The command runs on the local host that launched `helix`, even when the optimize request itself uses remote execution.
- The command is executed through the system shell so callers can use normal shell syntax and variable expansion.
- No new timeout is introduced for the post command.
- When `--stream-output` is enabled, the post-command stdout/stderr are surfaced through the same workspace-prefixed stream after the command completes.
- When `--verbose` is enabled, batch logs print a one-line "running" message before the command starts.
- If the post-optimize command exits non-zero, that workspace is reported as failed and its batch status entry is written as `incomplete`.
- Other workspaces keep running. One workspace's post command failure does not abort the whole batch executor.

## Execution Order

For each runnable workspace:

1. Run the existing optimize request.
2. If optimize fails, keep existing failure behavior and do not run the post command.
3. If optimize succeeds and `--post-optimize-command` is set, run the shell command in the workspace.
4. Only after the post command succeeds should batch-local follow-up work continue:
   - auto-upload
   - auto-report
   - batch status update to `completed`
   - final batch result status `ok`

This keeps the post command part of the success path for a workspace.

## Implementation Shape

- Extend `OptimizeRunOptions` with `post_optimize_command: str | None = None`.
- Add one CLI flag on `optimize-batch` only.
- Parse the batch-only flag into the shared options object, but only the batch runtime consumes it.
- Keep the command execution helper local to `src/helix/optimize/batch.py`.
- Execute the command through the system shell so callers can pass an ordinary command string.
- Reuse the existing batch failure-summary pattern: prefer the last non-blank stderr line, then stdout, then a return-code fallback.

## Testing

- Parser coverage proving `optimize-batch` accepts `--post-optimize-command` and stores it in `OptimizeRunOptions`.
- Runtime coverage proving a successful workspace optimize run triggers the post command with the workspace path.
- Runtime coverage proving a non-zero post command exit marks the workspace failed and leaves the batch status entry `incomplete`.
