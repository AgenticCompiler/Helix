# Codex Ephemeral and Git Check Flags

## Summary

Launch Codex with `--skip-git-repo-check`, and use `--ephemeral` when the current workflow requests a non-persistent Codex session.

## User-Visible Behavior

- Codex launches should not require the target workspace to pass Codex's git repository check.
- Non-optimize non-interactive Codex runs keep ephemeral sessions.
- `optimize --no-agent-session` uses `--ephemeral` for Codex.

## Implementation Notes

- Keep `--skip-git-repo-check` on the non-interactive `codex exec` command.
- Apply `--ephemeral` only where the current workflow requires a non-persistent Codex session.
- Cover the relevant command shapes with unit tests so future command-builder changes preserve the intended optimize-specific behavior.
