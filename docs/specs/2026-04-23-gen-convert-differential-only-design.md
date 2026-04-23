# Gen Convert Differential-Only Design

## Summary

- Redefine `gen-convert` as a convert-only workflow with differential correctness validation.
- Remove baseline preparation and benchmark generation from the `gen-convert` contract.
- Treat the original PyTorch operator as source material and correctness oracle for the generated differential test.
- Keep `optimize` as the only workflow that establishes or reuses canonical `baseline/` artifacts.

## Problem

The current `gen-convert` design mixes two different goals:

1. convert one PyTorch operator into a Triton NPU-backed operator
2. prepare optimize-style baseline artifacts for the converted output

That coupling is the wrong product boundary.

`convert` and `optimize` have different validation semantics:

- `optimize` needs a reusable canonical baseline because it measures performance deltas across later rounds
- `convert` only needs to prove that the converted operator is correct relative to the original PyTorch operator

Building `baseline/` for `gen-convert` adds optimize-specific state and benchmark obligations that do not serve the conversion workflow itself. It also makes the command contract misleading by teaching users that conversion is an optimize pre-step rather than a correctness-validated artifact generation task.

## Goals

- Make `gen-convert` produce one converted operator artifact and validate it through differential testing.
- Use the original PyTorch operator as the correctness oracle for conversion validation.
- Keep the original input file unchanged.
- Remove benchmark and baseline responsibilities from the `gen-convert` user-visible contract.
- Align README, prompts, staged skills, and tests with the narrower convert semantics.

## Non-Goals

- Do not change optimize baseline behavior.
- Do not weaken optimize requirements around `baseline/`, benchmarking, or round comparisons.
- Do not introduce a new convert-specific baseline or performance artifact format.
- Do not make `gen-convert` run in-place optimization rounds or create `opt-round-*` directories.
- Do not support standalone correctness mode for conversion validation in this redesign.

## User-Facing Command Contract

### Purpose

`gen-convert` should mean:

- read one original PyTorch operator file
- write one new Triton NPU-backed PyTorch operator file
- generate and execute one differential test that compares original-versus-converted outputs

The command should no longer imply benchmark preparation or optimize-session setup.

### Inputs

- one original PyTorch operator file
- optional explicit output path for the converted operator
- optional remote execution context

### Outputs

- one converted operator file, defaulting to `triton_<stem>.py`
- one generated differential test file
- archived correctness-validation output from differential test execution when the existing run-eval workflow produces it
- a short summary of the conversion result and any blockers

### Completion Condition

`gen-convert` is complete only when:

- the converted operator file exists at the requested output path
- a differential test exists for the converted operator
- the differential test has been executed against the converted operator
- the converted operator matches the original PyTorch operator behavior within the existing differential-test contract

If the differential test cannot be made to pass, the workflow should stop with a clear correctness or environment blocker instead of creating baseline-style fallback artifacts.

## Validation Semantics

### Oracle And Target

For `gen-convert`, the validation roles should be:

- original input operator: source material and correctness oracle
- converted output operator: system under test

The workflow should never construct a baseline from the converted operator just to compare the converted operator back to itself later.

### Test Mode

`gen-convert` should support only `differential` validation semantics.

That means:

- the CLI should not expose `standalone` as a meaningful convert mode
- prompts and skills should assume differential comparison by default
- generated convert validation should compare original-versus-converted behavior, not only assert local invariants on the converted file

If the parser continues to accept `--test-mode` for consistency with other generation commands, it should reject any value other than `differential` with a short actionable error.

### Benchmark Removal

`gen-convert` should not:

- accept `--bench-mode`
- generate a benchmark harness
- execute a benchmark
- claim completion based on benchmarkability

Performance exploration remains the responsibility of `optimize`, not `gen-convert`.

## Skill And Prompt Boundary

### Staged Skills

`gen-convert` should stage only the skills needed for:

- conversion
- differential test generation
- differential test execution
- operator-side repair when conversion hits Triton-side errors

It should stop staging optimize-only or benchmark-only skills such as:

- `triton-npu-prepare-optimize-baseline`
- `triton-npu-gen-bench`
- `triton-npu-optimize-check`

### Convert Skill Contract

`triton-npu-convert-pytorch-operator` should be rewritten to say:

- the original input operator is source material and oracle
- the converted output must remain PyTorch-facing and Triton NPU-backed
- the trailing input-helper block should still be preserved in the converted output when present
- the workflow must generate and run a differential test against the converted output
- the workflow finishes after correctness passes or a clear blocker is reported

It should no longer say:

- establish `baseline/`
- build reusable baseline artifacts
- validate benchmarkability
- delegate to optimize-baseline preparation

### Prompt Contract

The `gen-convert` prompt should explicitly require:

- do not execute the original input file as the converted artifact under validation
- use the original operator as the differential reference implementation
- validate the converted output through differential testing
- do not create `baseline/`
- do not generate or run benchmark artifacts

This keeps prompt wording aligned with the command's product meaning.

## Artifact Contract

### Expected Workspace Artifacts

After a successful `gen-convert` run, the workspace may contain:

- the converted operator file
- a generated `differential_test_<name>.py` file
- archived differential result payloads from test execution

These are sufficient for the convert workflow.

### Removed Artifacts

`gen-convert` should no longer create or require:

- `baseline/`
- `baseline/state.json`
- `baseline/perf.txt`
- generated benchmark harnesses
- optimize round directories

This separation keeps optimize session state out of conversion workspaces unless the user later runs `optimize`.

## CLI And Runtime Changes

### Parser Surface

Update `gen-convert` so that:

- `--output` remains supported
- `--agent`, `--interact`, `--show-output`, `--force-overwrite`, `--remote`, and `--remote-workdir` remain supported
- `--test-mode` is fixed to `differential` semantics
- `--bench-mode` is removed from the command

### Generation Orchestration

The generation request path for `gen-convert` should:

- resolve the converted output path as before
- stage only convert-relevant skills
- build a prompt that requires differential correctness validation only
- avoid passing benchmark expectations into the convert contract

No optimize orchestration changes are required beyond removing the accidental convert-to-baseline coupling from the convert workflow.

## Documentation Impact

Update user-facing docs so they consistently describe `gen-convert` as:

- conversion of one original PyTorch operator into one Triton NPU-backed operator
- preservation of the trailing input-helper block
- differential correctness validation against the original operator

Remove statements that say or imply:

- `gen-convert` prepares `baseline/`
- `gen-convert` generates benchmark harnesses
- `gen-convert` leaves behind reusable optimize baseline artifacts

README examples and command summaries should reflect the narrower contract.

## Testing Impact

Tests should cover:

- parser behavior for `gen-convert` without `--bench-mode`
- rejection of unsupported convert test modes when applicable
- staged-skill allowlist for `gen-convert`
- prompt text for differential-only convert semantics
- README and skill contract wording so baseline and benchmark language does not reappear

Existing optimize tests should remain responsible for baseline behavior.

## Migration Notes

This is a contract-correction change rather than a new feature family.

Implementation should focus on:

- shrinking the `gen-convert` CLI surface
- rewriting convert prompt and skill text
- updating orchestration tests and doc-contract tests
- removing convert-specific baseline and benchmark expectations from README and test fixtures

No compatibility shim is needed for old convert semantics because the old semantics are conceptually wrong for the intended workflow.

## Open Questions Resolved

- `gen-convert` should validate correctness through differential testing against the original PyTorch operator.
- `gen-convert` should not build or reuse optimize-style baseline artifacts.
- `gen-convert` should not generate or run benchmark harnesses.
- `optimize` remains the only workflow that owns canonical `baseline/` preparation and performance comparison state.
