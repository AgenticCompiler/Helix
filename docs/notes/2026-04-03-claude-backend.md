# Claude Backend

## Summary

Add `claude` as a fourth code-agent backend alongside `codex`, `opencode`, and `pi`.

## User-Visible Behavior

- Users can select `--agent claude` on the existing agent-backed commands.
- The CLI surface stays unchanged apart from allowing `claude` in the existing `--agent` choice list.
- In interactive mode, the CLI should launch `claude` in the target workspace with the generated prompt as the initial message.
- In non-interactive mode, the CLI should use `claude --print` with the generated prompt and return the child process exit code.
- Claude should discover workspace-local project skills from `.claude/skills/<name>/SKILL.md`.
- `optimize --no-agent-session` should request `claude --no-session-persistence` only in non-interactive mode; interactive Claude runs should ignore that option because the CLI does not expose an equivalent interactive flag.

## Implementation Notes

- Add a `ClaudeRunner` backend adapter that mirrors the existing runner responsibilities:
  - build backend-specific commands
  - select interactive, buffered, or streaming execution mode
  - emit verbose launch logging through the shared verbose helpers
- Keep the shared subprocess execution path in `process_runner.py`.
- Extend `SkillLinkManager` with Claude-specific staging under `.claude/skills`.
- Run Claude in the target workspace directory so its automatic project-skill discovery picks up the staged `.claude/skills` tree.
- For non-interactive Claude launches, pass `--dangerously-skip-permissions` so the wrapper stays script-friendly like the other non-interactive backends.
- Keep optimize resume behavior aligned with the other backends by reusing the existing continuation prompt pattern.

## Test Plan

- Add parser coverage showing `--agent claude` is accepted on agent-backed commands.
- Add runner unit tests covering:
  - interactive Claude command construction
  - non-interactive Claude command construction
  - optimize `--no-agent-session` behavior
  - verbose launch logging
  - shared process-runner dispatch
- Add skill staging tests covering `.claude/skills` copy staging and cleanup.
- Update user-facing docs and durable project rules to mention `claude`.
- Run:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
