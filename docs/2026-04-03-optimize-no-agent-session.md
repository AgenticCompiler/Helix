# Optimize No-Agent-Session Option

## Summary

Add an optimize-only `--no-agent-session` CLI option that asks supported code-agent backends to avoid retaining an agent session for that optimize run.

## User-Visible Behavior

- `optimize` accepts a new `--no-agent-session` flag.
- The flag affects only backend launch semantics; prompts, skill staging, and optimize workflow behavior stay unchanged.
- Backend behavior is backend-specific:
  - Codex uses `--ephemeral`
  - Pi uses `--no-session`
  - OpenCode ignores the flag because it does not currently expose a matching session-disabling option
- The flag applies only to `optimize`; generation commands keep their existing backend launch behavior.

## Implementation Notes

- Extend `AgentRequest` with an optimize-session control field so the CLI can pass the parsed option into runner adapters and resume flows.
- Keep parsing scoped narrowly to the `optimize` subcommand.
- Keep command construction isolated inside each backend adapter:
  - Codex toggles `--ephemeral` only for optimize runs when `--no-agent-session` is set
  - Pi toggles `--no-session` only for optimize runs when `--no-agent-session` is set
  - OpenCode ignores the field
- Update prior Pi and Codex backend docs so optimize-specific session handling no longer conflicts with the new option.

## Test Plan

- Add parser coverage for `optimize --no-agent-session`.
- Add CLI plumbing tests confirming the request passed into optimize supervision records the new option.
- Add backend command tests confirming:
  - Codex optimize toggles `--ephemeral`
  - Pi optimize toggles `--no-session`
  - OpenCode optimize ignores the option
- Run:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
