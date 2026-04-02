---
name: optimize
description: Iteratively optimize a Triton Ascend NPU operator with correctness and performance gates. Use for operator optimization tasks that need repeated correctness validation, benchmark validation, multi-round experiment tracking, reusable optimization notes, and profiler-backed performance analysis when benchmark results need deeper explanation.
---

# Optimize

Optimize one Triton Ascend NPU operator through repeated validated rounds.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

## Inputs

- Operator source code or an operator file path

## Outputs

- A sequence of `opt-round-N/` directories under the operator workspace
- One optimized operator file inside each completed round directory
- One `attempts.md` inside each round directory, updated throughout the round
- One `summary.md` inside each completed round directory
- Updated `opt-note.md` in the operator workspace
- Correctness and benchmark evidence produced through the `run-validate` skill

## Required Preconditions

- Ensure the operator workspace has correctness tests. If not, generate them with `test-gen` skill.
- Ensure the operator workspace has benchmark cases. If not, generate them with `bench-gen` skill.
- Do not start optimization rounds until both validation artifacts exist.

## Required References

- Read [workflow.md](references/workflow.md) before starting the first optimization round.
- Read [artifacts.md](references/artifacts.md) before creating any round directory or updating `opt-note.md`.
- Read [opt-note-format.md](references/opt-note-format.md) before editing `opt-note.md`.
- Read [contracts.md](references/contracts.md) when correctness or benchmark validation fails.
- Read [patterns/index.md](references/patterns/index.md) before choosing any optimization pattern reference.
- Use the sibling `ascend-npu-operator-profiler` skill when benchmark numbers need operator-level performance evidence, hotspot diagnosis, bottleneck analysis, or profiler-backed comparison across runs.
- Use the bundled helper script at [`../run-validation/scripts/run-command.py`](../run-validation/scripts/run-command.py) for generation, validation, and comparison commands; if the outer optimize task is remote-aware, carry the same remote flags through those commands.
- Treat `references/knowledge/` as optional background material for future expansion, not part of the minimum optimize workflow.

## Pattern References

Do not read all pattern references at once.

Use [patterns/index.md](references/patterns/index.md) to choose the most relevant optimization direction first. Treat that index as the single entry point for detailed pattern references under `references/patterns/`.

## Workflow

1. Inspect the operator workspace, resolve the correctness and benchmark modes, and confirm which validation artifacts already exist.
2. Generate missing tests or benchmarks through `../run-validation/scripts/run-command.py` before starting any optimization round.
3. Treat the original operator as validated candidate `round 0`, then choose one validated parent candidate for the next round instead of assuming the current best version is always the right parent.
4. Create `opt-round-N/`, copy the chosen parent operator into it, and start `attempts.md` immediately so every meaningful attempt and measurement is recorded.
5. Read `references/patterns/index.md`, pick one optimization hypothesis, and read only the one or two detailed pattern references that match that hypothesis.
6. Apply one coherent optimization theme for the round, then run correctness validation before trusting any performance result.
7. After correctness passes, run benchmark validation; when benchmark timing alone does not explain the result well enough, use `ascend-npu-operator-profiler` to gather operator-level evidence.
8. Use the benchmark and profiler evidence to decide whether to keep iterating, abandon the direction, or finalize the round `summary.md` and update `opt-note.md`.

## Quality Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Always run correctness before trusting performance results.
- Use profiler evidence when benchmark timing alone does not explain the result well enough.
- Keep parent-child traceability explicit so later engineers can understand which idea produced each round.
- Prefer multiple diverse optimization directions over a single greedy chain from the current best version.
- Prefer selective pattern reading over bulk-loading all optimization references.
- Do not silently discard optimization intent; preserve important comments that explain why a change helps.
- Record within-round attempts continuously so long-running rounds do not lose intermediate learning.
- Record optimization points in enough detail that another engineer could reuse them on a related operator.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
