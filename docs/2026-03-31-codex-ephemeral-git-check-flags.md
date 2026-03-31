# Codex Ephemeral and Git Check Flags

## Summary

Launch Codex with `--ephemeral` and `--skip-git-repo-check` in both interactive and non-interactive modes.

## User-Visible Behavior

- `--agent codex` runs should always start with ephemeral Codex sessions.
- Codex launches should not require the target workspace to pass Codex's git repository check.
- The CLI surface stays unchanged; this is backend launch behavior, not a new user-facing CLI option.

## Implementation Notes

- Add both flags to the interactive Codex TUI command.
- Add both flags to the non-interactive `codex exec` command.
- Cover both command shapes with unit tests so future command-builder changes preserve the flags.
