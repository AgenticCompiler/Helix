# Pi Backend

## Summary

Add `pi` as a third code-agent backend alongside `codex` and `opencode`.

## User-Visible Behavior

- Users can select `--agent pi` on the existing agent-backed commands.
- The CLI surface stays unchanged apart from allowing `pi` in the existing `--agent` choice list.
- In interactive mode, the CLI should launch `pi` with `--thinking high` and `--no-extensions`, using the generated prompt as the initial message.
- In non-interactive mode, the CLI should use `pi --print --thinking high --no-extensions` with the generated prompt and return the child process exit code.
- `optimize --no-agent-session` additionally requests Pi's `--no-session` mode.
- The repository `skills/` content should be staged into the target workspace in a Pi-specific location before launch so Pi reads workspace-local skill copies instead of repository source paths.

## Implementation Notes

- Add a `PiRunner` backend adapter that mirrors the existing runner responsibilities:
  - build backend-specific commands
  - select interactive, buffered, or streaming execution mode
  - emit verbose launch logging through the shared verbose helpers
- Keep the shared subprocess execution path in `process_runner.py`.
- Extend `SkillLinkManager` with Pi-specific staging under `.pi/skills`.
- For Pi launches:
  - set `--thinking high`
  - disable extension discovery with `--no-extensions`
  - pass the staged Pi skill path explicitly with `--skill`
  - disable implicit skill discovery with `--no-skills` so the repository skills remain the source of truth
- For optimize launches with `--no-agent-session`, add `--no-session`.
- Keep optimize resume behavior aligned with the other backends by reusing the existing continuation prompt pattern.

## Test Plan

- Add parser coverage showing `--agent pi` is accepted on agent-backed commands.
- Add runner unit tests covering:
  - interactive Pi command construction
  - non-interactive Pi command construction
  - verbose launch logging
  - shared process-runner dispatch
- Add skill staging tests covering `.pi/skills` copy staging and cleanup.
- Update user-facing docs and durable project rules to mention `pi`.
- Run:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
