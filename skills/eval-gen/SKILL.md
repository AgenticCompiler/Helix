---
name: eval-gen
description: Use when one code agent task should repair a Triton Ascend operator when needed, then generate and validate both correctness-test and benchmark harnesses.
---

# Eval Gen

Complete one combined evaluation-generation workflow for a single operator.

Use this skill when the user wants one code-agent task to:

- inspect an operator file
- repair the original operator file when needed
- generate a correctness test through `test-gen`
- generate a benchmark through `bench-gen`
- validate both artifacts through `operator-eval`

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
3. Generate the correctness test with the `test-gen` skill.
4. Validate the generated test with the `operator-eval` workflow, using the generated file against the current operator file.
5. If test validation fails, decide whether the failure belongs to the generated test or the operator:
   - repair the generated test when the harness is at fault
   - repair the original operator file when the operator is at fault
6. Re-run test validation after every relevant repair until the test passes or an environment blocker prevents progress.
7. Generate the benchmark with the `bench-gen` skill.
8. Validate the generated benchmark with the `operator-eval` workflow.
9. If benchmark validation fails, decide whether the failure belongs to the generated benchmark or the operator:
   - repair the generated benchmark when the harness is at fault
   - repair the original operator file when the operator is at fault
10. Re-run benchmark validation after every relevant repair until the benchmark passes or an environment blocker prevents progress.
11. Before finishing, confirm that both generated artifacts pass against the final operator file state.

## Validation Commands

- Use `python3 ../operator-eval/scripts/run-command.py run-test ...` for correctness validation.
- Use `python3 ../operator-eval/scripts/run-command.py run-bench ...` for benchmark validation.
- If the outer task is remote-aware, carry the same remote flags into every validation command and reuse `--remote-workdir` when provided.

## Quality Rules

- Repair the original operator file when the operator is the source of failure.
- Keep generated test and benchmark files aligned with the final operator API.
- Prefer targeted repairs over broad rewrites.
- Stop with a short explicit explanation when the problem is a workspace or environment blocker that cannot be fixed from repository code alone.

## Do Not

- Do not create `opt-round-*` directories.
- Do not create or update `opt-note.md`.
- Do not use the `optimize` skill for this workflow.
- Do not use profiler or IR-analysis flows for this workflow.
- Do not leave either generated artifact unvalidated.
