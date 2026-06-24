---
name: ascend-npu-gen-eval-suite
description: Use when one code agent task should repair a Triton Ascend operator when needed, then generate and validate both correctness-test and benchmark harnesses.
---

# Eval Gen

Complete one combined evaluation-generation workflow for a single operator.

Use this skill when the user wants one code-agent task to:

- Generate a correctness test via `ascend-npu-gen-test`.
- Generate a benchmark via `ascend-npu-gen-bench`.
- Validate both artifacts via `ascend-npu-run-eval`.

## Inputs

- One operator file path
- One requested test mode: `standalone` or `differential`
- One requested benchmark mode: `standalone` or `msprof`
- Optional remote execution context from the outer task

## Outputs

- A repaired operator file when operator-level fixes are required
- One generated correctness test file
- One generated benchmark file
- A short summary describing operator repairs, generated artifacts, and any remaining environment blockers

## Required Workflow

1. Read the operator file and identify whether the operator itself already has clear correctness, signature, import, or runtime issues.
2. If the operator is clearly at fault, repair the original operator file directly before generating harnesses.
3. Ensure the operator’s Triton path actually runs, and remove any code paths that fall back to PyTorch.
4. Generate the correctness test with the `ascend-npu-gen-test` skill.
5. Validate the generated test with the `ascend-npu-run-eval` workflow, using the generated file against the current operator file.
6. If test validation fails, decide whether the failure belongs to the generated test or the operator:
   - repair the generated test when the harness is at fault
   - repair the original operator file when the operator is at fault
7. Re-run test validation after every relevant repair until the test passes or an environment blocker prevents progress.
8. Generate the benchmark with the `ascend-npu-gen-bench` skill.
9. Validate the generated benchmark with the `ascend-npu-run-eval` workflow.
10. If benchmark validation fails, decide whether the failure belongs to the generated benchmark or the operator:
   - repair the generated benchmark when the harness is at fault
   - repair the original operator file when the operator is at fault
11. Re-run benchmark validation after every relevant repair until the benchmark passes or an environment blocker prevents progress.
12. Before finishing, confirm that both generated artifacts pass against the final operator file state.

## Repair experience (Ascend Triton)

When steps 2, 6, or 10 require **operator-side** fixes for compile / JIT / kernel errors on Ascend, use the `triton-npu-repair-guide` skill and read its `references/repair-experience.md` for team-maintained heuristics. These hints do not override `ascend-npu-gen-test`, `ascend-npu-gen-bench`, or normative specs.

If the successful fix is a new pattern not covered there, append a short entry to the `triton-npu-repair-guide` skill's `output.md`.

## Validation Commands

- Use the `ascend-npu-run-eval` skill for correctness validation, with `python3 ../ascend-npu-run-eval/scripts/run-command.py run-test-baseline ...` as the standard helper command.
- Use the `ascend-npu-run-eval` skill for benchmark validation, with `python3 ../ascend-npu-run-eval/scripts/run-command.py run-bench ...` as the standard helper command.
- If the outer task is remote-aware, carry the same remote flags into every validation command and reuse `--remote-workdir` when provided.

## Quality Rules

- Repair the original operator file when the operator is the source of failure.
- Keep generated test and benchmark files aligned with the final operator API. They must follow **`ascend-npu-gen-test` / `ascend-npu-gen-bench`** norms: harnesses **run on Ascend NPU only**—no CUDA/CPU/other-device primary paths (see those skills’ specs).
- If either generated harness uses randomized inputs, it must explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.
- Prefer targeted repairs over broad rewrites.
- Stop with a short explicit explanation when the problem is a workspace or environment blocker that cannot be fixed from repository code alone.
- When editing the operator, **preserve the Triton / NPU kernel path** as the delivered implementation. **Do not** replace it with a **pure PyTorch** reimplementation as a “fix,” and **do not** satisfy validation by weakening what the harness exercises.

## Do Not

- Do not mask operator failures in harnesses. Never add try/except, alternate paths, shims, or PyTorch-only fallbacks in generated tests/benchmarks to green `ascend-npu-run-eval` correctness-test / `run-bench` validations when the real issue is Triton compile, launch, or operator logic—**forbidden**.
- Do not patch only the harness for compiler/kernel failures
- Do not create `opt-round-*` directories.
- Do not create or update `opt-note.md`.
- Do not use the `triton-npu-optimize` skill for this workflow.
- Do not use profiler or IR-analysis flows for this workflow.
- Do not leave either generated artifact unvalidated.
