# Stream Output Default And Git Fallback Design

## Summary

- Replace the CLI surface flag `--show-output` with `--no-stream-output`.
- Make non-interactive agent commands stream output by default.
- Change optimize batched round default size from `10` to `5`.
- Skip temporary workspace git initialization when `git` is not installed locally.

## Goals

- Simplify the CLI so streaming is the default behavior instead of an opt-in.
- Keep existing runtime internals stable by preserving the internal `show_output` request field.
- Reduce optimize worker batch size default to `5`.
- Avoid failing agent startup on machines without a `git` executable.

## Non-Goals

- Do not rename existing `show-output.log` files or internal `show_output` model fields.
- Do not change interactive mode behavior.
- Do not redesign skill staging or git cleanup beyond the missing-`git` fallback.

## User-Visible Behavior

- Agent-backed non-interactive commands now stream output unless the user passes `--no-stream-output`.
- `--show-output` is removed from the CLI surface and should no longer parse.
- `optimize` and `optimize-batch` now default `--round-batch-size` to `5`.
- If skill staging needs a temporary git boundary and `git` is unavailable in `PATH`, staging continues without creating `.git`.

## Implementation Notes

- Keep the internal boolean named `show_output`, but map it from `not args.no_stream_output`.
- Update parser defaults so agent-backed commands set `show_output=True` when no flag is passed.
- Leave renderers, runners, and log writers on the existing `show_output` contract.
- In `SkillLinkManager`, check `shutil.which("git")` before attempting `git init`.
- When `git` is missing, return `None` for the temporary git dir and continue staging normally.

## Testing

- Parser coverage for default streaming on and `--no-stream-output` disabling it.
- Parser coverage proving `--show-output` no longer parses.
- Optimize parser and model coverage for the new round batch default of `5`.
- Skill staging coverage proving missing `git` skips temp repo creation without failing.
