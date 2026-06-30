# Run-Eval Skill Helper Path Design

## Goal

Remove ambiguous `./scripts/run-command.py` guidance from the live
`ascend-npu-run-eval` skill docs so agents stop resolving the helper script
relative to the current working directory instead of the staged skill root.

## User-Visible Semantics

- Live `ascend-npu-run-eval` skill instructions and examples should refer to the
  helper entrypoint as `python3 <skill-path>/scripts/run-command.py ...`.
- The docs should describe `<skill-path>` as the staged path of the current
  skill for the active backend, rather than implying that `./scripts/` exists
  under the agent's current working directory.
- This change is documentation-contract only. It does not change helper script
  behavior, backend staging layout, or repository CLI entrypoints.

## Problem

The current live skill text still documents commands like:

```bash
python3 ./scripts/run-command.py run-bench ...
```

That shape is only valid when the shell happens to be running from the skill
root itself. In real agent runs, the working directory is usually the operator
workspace, so the same command incorrectly targets a workspace-local
`./scripts/run-command.py` path or fails with a missing-file error.

The repository already documents staged helper execution as an explicit
skill-owned path in other places, and the backend guard design explicitly allows
commands shaped like `python3 .codex/skills/.../scripts/run-command.py ...` or
`python3 .opencode/skills/.../scripts/run-command.py ...`.

## Design

Update the live `skills/common/ascend-npu-run-eval/` documentation set so every
agent-facing helper invocation uses the backend-neutral placeholder form:

```bash
python3 <skill-path>/scripts/run-command.py <subcommand> ...
```

Apply that wording to:

- `SKILL.md`
- `references/run-test.md`
- `references/run-bench.md`
- `references/probe-bench.md`
- `references/profile-bench.md`
- `references/profile-report.md`
- `references/compare-result.md`
- `references/compare-perf.md`

The top-level skill doc should also explain what `<skill-path>` means in one
short sentence so dependent skills can reference the run-eval surface without
reintroducing backend-specific hard-coded staged paths.

## Test Contract

Extend `tests/test_generation_contracts.py` so it asserts the live
`ascend-npu-run-eval` docs:

- contain `<skill-path>/scripts/run-command.py`
- do not contain `python3 ./scripts/run-command.py`

This keeps future doc edits from regressing to current-directory-relative helper
paths.

## Non-Goals

- Do not change the helper script implementation under `scripts/`.
- Do not change MCP skill wording in this task.
- Do not hard-code `.codex/skills/` or `.opencode/skills/` into the live skill
  docs.
