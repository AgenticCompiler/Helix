# Optimize `--system-prompt` Memory File Append

## User-visible behavior

Add a new `--system-prompt` option to `helix optimize` and `helix optimize-batch`.

- When the value is plain text, append that text to the temporary optimize memory file (`AGENTS.md` or `CLAUDE.md`) after the built-in guidance.
- When the value starts with `@`, treat the remainder as a file path, read that file as UTF-8 text, and append the file contents instead.
- The appended content is only for the temporary optimize memory file created for the run. It does not overwrite the user’s original memory file permanently.
- The existing `--prompt` behavior is unchanged. `--prompt` still feeds worker prompts; `--system-prompt` only affects the temporary memory file.

## Path and error semantics

- `@path` resolves relative paths against the current process working directory, matching normal CLI expectations.
- Missing or unreadable `@path` inputs should fail argument handling with a short actionable parser error.
- An empty `--system-prompt` value, or a referenced file whose contents are empty after trimming, is treated as no extra appended block.

## Implementation shape

- Parse and resolve `--system-prompt` in the optimize command path before building `OptimizeRunOptions`.
- Store the resolved text on optimize request/options state so the session-artifact layer can pass it into memory-file rendering.
- Extend optimize memory-file rendering with one optional trailing block dedicated to user-supplied system guidance.

## Verification

- CLI parsing test for `optimize` plain-text `--system-prompt`.
- CLI parsing test for `optimize-batch` plain-text `--system-prompt`.
- Main-command test proving `@path` content reaches the optimize request.
- Guidance rendering tests proving appended content is present in both checked and supervised temporary memory files.
