# Codex Agent Hook Read Coverage Design

## Goal

Close the two confirmed escape paths in the opt-in Codex agent hook guard:

- direct Codex `Read` tool access to protected staged skill scripts
- nested shell reads such as `bash -lc "sed -n ... .codex/skills/.../scripts/..."`

## User-Visible Behavior

- `optimize --enable-agent-hooks --agent codex` should deny direct reads of protected staged skill scripts when Codex uses the built-in `Read` tool.
- The same run should also deny protected reads hidden behind shell wrapper commands such as `bash -c`, `bash -lc`, `sh -c`, and `zsh -c`.
- Allowed reads inside the workspace that do not target protected paths should continue to work.
- The denial message should stay unchanged so existing guidance remains stable.

## Design

### Hook Registration

Extend the staged Codex `hooks.json` so the same `PreToolUse` guard runs for both:

- `Bash`
- `Read`

This keeps the current workspace-local hook staging model and avoids introducing a second guard implementation just for direct file reads.

### Guard Evaluation

Keep a single `pretooluse_guard.py` entrypoint, but expand it to evaluate two tool families:

- `Bash`: continue checking read-oriented shell commands, and recursively inspect the command string passed to `-c` or `-lc` wrapper shells.
- `Read`: inspect the requested file path from the tool input and apply the same workspace and deny-glob policy checks.

For `Read`, the implementation should accept the field names we already know about from local hook usage:

- `file_path`
- `filePath`

### Shell Wrapper Parsing

The current guard only inspects top-level shell tokens, which misses wrapped commands. The updated guard should:

- tokenize the command as it does today
- treat `bash`, `sh`, and `zsh` as wrapper shells
- when one of those wrappers is followed by `-c` or `-lc`, recursively analyze the following command string
- merge candidate paths discovered in the wrapped command into the normal evaluation flow

This remains a conservative parser. It should cover the confirmed `bash -lc` bypass without attempting to model all shell syntax.

## Testing

Add focused unit coverage for:

- direct `Read` access to `.codex/skills/*/scripts/**`
- direct `Read` access outside the workspace
- nested `bash -lc` reads of protected staged skill scripts
- staged `hooks.json` containing both `Bash` and `Read` matchers

## Scope Boundaries

- Do not expand protection from `scripts/` to all staged skill files.
- Do not add global Codex configuration changes.
- Do not broaden this into a general sandbox or arbitrary shell parser.
