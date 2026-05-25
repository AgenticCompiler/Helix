# Codex Agent Hook Read Coverage Implementation Plan

## Goal

Make the opt-in Codex agent hook block protected staged skill script reads through both the built-in `Read` tool and nested shell wrappers such as `bash -lc`.

## Steps

1. Add failing tests in `tests/test_codex_pretooluse_guard.py` for direct `Read` access and nested `bash -lc` protected reads.
2. Add a staging assertion in `tests/test_agent_hooks.py` that `hooks/codex/hooks.json` is copied with both `Bash` and `Read` matchers.
3. Update `hooks/codex/hooks.json` to register the guard for `Read` in addition to `Bash`.
4. Update `hooks/codex/pretooluse_guard.py` so it:
   - evaluates `Read` tool payloads via `file_path` or `filePath`
   - recursively inspects `bash|sh|zsh -c|-lc` wrapped commands
5. Run focused unit tests for the hook manager and guard.
6. Update `README.md` and any wording that still describes Codex coverage as Bash-only.
