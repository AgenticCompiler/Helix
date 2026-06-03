# Gen Convert Design

## Summary

- Add a new generation-style subcommand: `gen-convert`.
- Add a new skill: `triton-npu-convert-pytorch-operator`.
- The command should stage that conversion skill plus the baseline-building skills it depends on, then launch a code agent.
- The conversion workflow should treat the input operator as source material only, write the converted operator to `triton_<origin-name>.py` by default, and establish a reusable `baseline/` for the converted output.
- `optimize` should stop staging the entire repo skill tree and instead stage an explicit optimize-only skill set that excludes the new conversion skill.

## Problem

The repository already supports:

- generating correctness harnesses
- generating benchmark harnesses
- combined evaluation setup
- iterative optimization

It does not yet support a first-class workflow for taking a PyTorch operator file and turning it into a PyTorch-facing operator backed by a Triton NPU kernel, while also producing a benchmarkable baseline for that converted output.

That gap matters because conversion has different rules from optimize:

- the original input operator should not be executed
- the output operator is a new artifact, not an in-place optimize round
- the trailing input-helper block in the input file must be preserved for later harness use
- the converted output must leave behind a reusable `baseline/`

## Goals

- Add one new user-facing CLI entrypoint for conversion.
- Keep the CLI thin and reuse the existing generation-style orchestration path.
- Make the conversion workflow skill-first.
- Preserve the original input operator file.
- Preserve the input file's trailing input-helper block in the converted output.
- Build baseline artifacts against the converted output operator.
- Keep `optimize` isolated from this new skill unless the user explicitly invokes conversion.

## Non-Goals

- Do not add a batch conversion command in this change.
- Do not move baseline-building logic from skills into the CLI.
- Do not change the optimize round contract itself.
- Do not require the conversion workflow to use only the provided trailing helpers when generating tests or benchmarks.

## User-Facing Behavior

The new command should look like:

```bash
uv run triton-agent gen-convert --input a.py
```

Default behavior:

- read `a.py` as the source operator
- write the converted operator to `triton_a.py`
- keep the original file unchanged
- preserve the input file's trailing input-helper block in the converted output
- use the converted output as the operator under validation
- establish `baseline/` for the converted output before finishing

Supported options should match the existing generation-style command surface where relevant:

- `--output`
- `--agent`
- `--interact`
- `--show-output`
- `--force-overwrite`
- `--remote`
- `--remote-workdir`
- `--test-mode`
- `--bench-mode`

## Skill Boundary

### New skill: `triton-npu-convert-pytorch-operator`

This skill should own:

- reading the original operator file
- deciding which PyTorch operators to replace with Triton NPU kernels
- writing the converted PyTorch-facing operator to the requested output path
- preserving the trailing input-helper block from the original file
- refusing to execute the original input operator file
- using `triton-npu-prepare-optimize-baseline` to establish a reusable baseline for the converted output

### Existing sibling skills

The conversion skill should delegate baseline work instead of re-describing it:

- `triton-npu-prepare-optimize-baseline`
- `triton-npu-gen-test`
- `triton-npu-gen-bench`
- `triton-npu-run-eval`
- `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`

## Runtime Design

- Extend `CommandKind` with `GEN_CONVERT`.
- Treat `gen-convert` as a generation command routed through `commands/generation.py`.
- Reuse `generation/orchestration.py` for request construction and runner lifecycle.
- Add a dedicated staged-skill allowlist for `gen-convert`.
- Add a dedicated optimize staged-skill allowlist so optimize no longer sees unrelated skills by default.
- Add prompt text that clearly distinguishes:
  - source operator input
  - requested converted output
  - "do not execute the original input operator"
  - "preserve trailing input-helper block"
  - "baseline must be built for the converted output"

## Output Naming

- Default converted output path: `triton_<stem>.py`
- This naming should be supported by `default_generated_output_path(...)`.
- `--output` should continue to override the default path.

## Validation And Tests

Tests should cover:

- parser support for `gen-convert`
- help output and command grouping
- default output naming to `triton_<stem>.py`
- restricted staged skill set for `gen-convert`
- optimize staged skill set excludes the new conversion skill
- skill-contract docs and README coverage for the new workflow

## Naming Decision

- Command: `gen-convert`
- Skill: `triton-npu-convert-pytorch-operator`

This keeps the CLI consistent with existing `gen-*` workflows while making the skill name explicit about both source format and target runtime path.
