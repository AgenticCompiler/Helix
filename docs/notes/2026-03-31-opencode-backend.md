# OpenCode Backend

## Summary

Add `opencode` as a second code-agent backend alongside `codex`.

## User-Visible Behavior

- Users can select `--agent opencode` on the existing commands.
- In interactive mode, the CLI should launch `opencode` in project mode with `--prompt`.
- In non-interactive mode, the CLI should use `opencode run --dir <workdir> <prompt>`.
- Skill copies for OpenCode should be exposed under `.opencode/skills/<name>/SKILL.md`.

## Implementation Notes

- Keep the shared subprocess execution path in `process_runner.py`.
- Add an `OpenCodeRunner` backend adapter for command construction and mode selection.
- Generalize the skill staging manager so it can prepare either Codex or OpenCode workspace skill directories.
