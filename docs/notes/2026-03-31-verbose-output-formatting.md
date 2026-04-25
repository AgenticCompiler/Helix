# Verbose Output Formatting

## Summary

Make verbose CLI output human-readable instead of dumping raw shell-like command lines.

## User-Visible Behavior

- Verbose output should group events by category with short prefixes such as `agent`, `skills`, and `files`.
- Prefixes may use color when the output stream supports it.
- Skill links should be shown in `a -> b` form so the link target is visible immediately.
- When `--force-overwrite` removes an existing generated file, verbose output should explicitly say so.
- Agent launch details should show the command and the prompt separately instead of embedding a long multiline prompt inside one shell-quoted line.

## Implementation Notes

- Keep formatting in a dedicated helper so CLI orchestration and backend runners can share it.
- Only apply ANSI color when writing to a TTY-capable stream.
- Preserve plain-text readability when color is not available, including in redirected output and tests.
