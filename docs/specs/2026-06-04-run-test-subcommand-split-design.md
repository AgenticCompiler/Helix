# Run-Test Subcommand Split Design

## Summary

- Add `run-test-baseline` and `run-test-optimize` to `skills/triton-npu-run-eval/scripts/run-command.py`.
- Keep the existing `run-test` entrypoint, but align it to the same baseline-result and baseline-operator-file semantics.
- Update skill guidance so agents are instructed to use only the new subcommands.
- Make optimize differential flows use baseline inputs instead of oracle-result naming.

## Problem

The current single `run-test` subcommand is used by both baseline-validation and optimize-validation workflows. Skill text must explain which flags belong to which workflow, and agents can still invoke the wrong shape of command even when the surrounding skill intent is clear.

This is especially risky for optimize flows in `differential` mode, where the workflow should always compare against a baseline payload. If the agent omits the required baseline input, the command still runs, but the optimize workflow loses the required correctness gate.

## Goals

- Encode baseline-vs-optimize intent directly in the helper CLI surface.
- Make skill guidance unambiguous so staged agents follow the right command by default.
- Reject optimize differential runs that do not provide the required baseline input.
- Preserve the existing execution implementation as much as possible.

## Non-Goals

- Do not rename the repository-level `triton-agent run-test` command in this change.
- Do not remove compatibility support for existing external callers of `run-command.py run-test`.
- Do not redesign benchmark, compare-result, or profile subcommands.

## Design

### New skill-side subcommands

Add two new `run-command.py` subcommands:

- `run-test-baseline`
- `run-test-optimize`

Both accept the same execution flags as the current helper `run-test` surface, with optimize-specific additions:

- `--test-file`
- `--operator-file`
- `--baseline-result`
- `--baseline-operator-file`
- `--compare-level`
- `--remote`
- `--remote-workdir`
- `--keep-remote-workdir`
- `--verbose`
- `--test-mode`

### Behavioral split

`run-test-baseline`

- Reuses the current `run-test` execution behavior.
- Allows standalone or differential execution.
- In differential mode, may optionally compare against `--baseline-result` or derive/auto-produce one from `--baseline-operator-file`.

`run-test-optimize`

- Reuses the same execution backend and output shape.
- In standalone mode, behaves like baseline execution.
- In differential mode, requires exactly one of:
  - `--baseline-result`
  - `--baseline-operator-file`
- If `--baseline-result` is provided, compare against it directly.
- If `--baseline-operator-file` is provided, derive the archived baseline payload path with the existing rule `<baseline-operator-stem>_result.pt` beside that operator file.
- If the derived baseline payload already exists, compare against it directly.
- If the derived baseline payload does not exist, automatically run the baseline test flow first to produce it, then compare.
- If `--test-mode` is omitted, the optimize-only requirement is decided after resolving metadata from the test file.

### Compatibility

Keep `run-test` in `run-command.py` as a compatibility alias for the old behavior. It should remain visible enough for existing callers to keep working, but all repository skill guidance should stop recommending it.

### Skill-documentation changes

Update `triton-npu-run-eval` and dependent skills so agent-facing instructions use:

- `run-test-baseline` for baseline or generation validation
- `run-test-optimize` for optimize-round validation

`compare-result` guidance should recommend `run-test-optimize --baseline-operator-file ...` for the one-command differential optimize path.

### Baseline output visibility

`run-test-baseline` should continue printing `Return code: ...` and, in differential mode, must print the archived `.pt` path after a successful run so downstream optimize flows and humans can see which payload was produced.

## Files

| File | Change |
|------|--------|
| `docs/specs/2026-06-04-run-test-subcommand-split-design.md` | Record behavior and compatibility contract |
| `skills/triton-npu-run-eval/scripts/run-command.py` | Add subparsers and optimize-only oracle enforcement |
| `skills/triton-npu-run-eval/SKILL.md` | Route test execution docs through the new command names |
| `skills/triton-npu-run-eval/references/run-test.md` | Replace with guidance for baseline and optimize subcommands |
| `skills/triton-npu-gen-test/SKILL.md` | Use `run-test-baseline` for generated-test validation |
| `skills/triton-npu-gen-eval-suite/SKILL.md` | Use `run-test-baseline` in generation flows |
| `skills/triton/triton-npu-repair-guide/SKILL.md` | Refer to baseline/optimize test execution explicitly |
| `tests/test_skill_command_script.py` | Cover parser/help/dispatch/error behavior |
| `tests/test_generation_contracts.py` | Lock new skill wording |
| `tests/test_codex_pretooluse_guard.py` | Accept staged helper invocation with new subcommands |
| `tests/test_opencode_hook_guard.py` | Accept staged helper invocation with new subcommands |
| `src/triton_agent/backends/codex_trace.py` | Classify new subcommands as correctness tests |
| `src/triton_agent/backends/claude_trace.py` | Classify new subcommands as correctness tests |

## Verification

- Run focused unit tests for the helper script, skill contract docs, and guard/trace behavior.
- Run `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py`.
