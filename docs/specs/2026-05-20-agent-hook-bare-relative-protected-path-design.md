# Agent Hook Bare Relative Protected Path Design

## Goal

Keep the recent staged-helper entrypoint fix while restoring protection for
normal bare relative paths under `triton-agent-logs/`.

## Problem

The current guards only recognize explicit path tokens that start with `/`,
`./`, `../`, or the backend-native staged skill root. After tightening the
fallback fragment regex, normal agent commands such as
`cat triton-agent-logs/gen-test.show-output.log` no longer produce any checked
candidate path, so the deny glob never runs.

## Design

- Treat `triton-agent-logs/` as a protected relative path prefix during shell
  token scanning.
- Extend the fallback fragment regex to recognize `triton-agent-logs/` inside
  quoted strings, so Python one-liners such as
  `python3 -c "print(open('triton-agent-logs/...').read())"` stay blocked.
- Keep the existing staged-helper entrypoint exception and nested-fragment
  filtering unchanged.

## Scope Boundaries

- Do not broaden generic bare relative path parsing beyond the known protected
  prefix.
- Do not revisit deliberate sandbox-evasion cases outside the current review
  scope.
