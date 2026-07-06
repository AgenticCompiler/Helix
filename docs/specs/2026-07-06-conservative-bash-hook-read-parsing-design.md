# Conservative Bash Hook Read Parsing Design

## Goal

Reduce false-positive Bash read denials in agent hooks. For this guard, a
missed read is acceptable; an incorrect denial is not.

## Problem

The current shared Bash guard scans an entire shell command once any
read-oriented command token appears anywhere in the token list. That broad scan
causes false positives:

- a later `head`, `grep`, or `tail` makes earlier non-read command arguments
  look like protected reads
- path-like substrings inside `python3 -c "..."` code are treated as file paths
- chained commands joined by `|`, `;`, `&&`, or `||` contaminate each other

These false positives block normal agent workflows.

## Design

- For Bash tool inputs, inspect only the first simple command.
- Preserve one layer of shell-wrapper unwrapping for `bash -c`, `bash -lc`,
  `sh -c`, and `zsh -lc`, then apply the same first-command rule to the nested
  command text.
- If the first simple command is not an explicit read command, allow the tool
  call without further shell parsing.
- Extract candidate paths only from that first simple command's argv tokens.
- Keep path detection conservative:
  - allow absolute paths
  - allow `./...` and `../...`
  - allow known protected relative prefixes such as `.triton-agent/`,
    `.codex/`, `.claude/`, `.opencode/`, and `triton-agent-logs/`
  - do not scan the full command string for embedded path fragments
  - do not parse `python -c`, heredoc bodies, or other embedded language text

## Expected Behavior

- `python3 .../cli.py --help 2>&1 | head -60` is allowed because the first
  simple command is `python3`, not a read command.
- `rm -rf ...; python3 -c "..."; grep -c ... /tmp/x; tail -5 /tmp/x` is
  allowed because the first simple command is `rm`, not a read command.
- Direct reads such as `cat triton-agent-logs/file.log` and
  `sed -n '1,20p' .codex/skills/.../scripts/cli.py` remain blocked.

## Scope Boundaries

- Do not attempt full shell parsing.
- Do not treat deliberate evasions hidden inside embedded code strings as in
  scope for this guard.
- Keep the built-in edit tool policy unchanged.
