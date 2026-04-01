---
name: optimize
description: Iteratively optimize a Triton Ascend NPU operator with correctness and performance gates. Use when Codex needs to improve an operator implementation, generate missing validation artifacts, run multi-round optimization experiments, record per-round summaries, and maintain reusable optimization notes for later engineers.
---

# Optimize

Optimize one Triton Ascend NPU operator through repeated validated rounds.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

## Inputs

- Operator source code or an operator file path
- Optional requested destination for a final promoted optimized file
- Optional requested correctness validation mode
- Optional requested benchmark validation mode
- Optional note that the outer wrapper is running in an interactive session

## Outputs

- A sequence of `opt-round-N/` directories under the operator workspace
- One optimized operator file inside each completed round directory
- One `attempts.md` inside each round directory, updated throughout the round
- One `summary.md` inside each completed round directory
- Updated `opt-note.md` in the operator workspace
- Correctness and benchmark evidence produced through `run-test` and `run-bench`

## Required Preconditions

- Ensure the operator workspace has correctness tests.
- Ensure the operator workspace has benchmark cases.
- Default to `differential` correctness validation unless the outer request specifies another mode.
- Default to `standalone` benchmark validation unless the outer request specifies another mode.
- If tests are missing, generate them with `test-gen` using the resolved correctness mode.
- If benchmarks are missing, generate them with `bench-gen` using the resolved benchmark mode.
- If the outer request explicitly specifies test or benchmark modes for optimization, keep the workflow aligned with those resolved modes and reflect them in the round guidance.
- Do not start optimization rounds until both validation artifacts exist.

## Required References

- Read [workflow.md](references/workflow.md) before starting the first optimization round.
- Read [artifacts.md](references/artifacts.md) before creating any round directory or updating `opt-note.md`.
- Read [opt-note-format.md](references/opt-note-format.md) before editing `opt-note.md`.
- Read [contracts.md](references/contracts.md) when correctness or benchmark validation fails.
- Read [patterns/index.md](references/patterns/index.md) before choosing any optimization pattern reference.
- Use the bundled helper script at [`../scripts/run-command.py`](../scripts/run-command.py) whenever the skill needs to execute project subcommands from this repository.
- Treat `references/knowledge/` as optional background material for future expansion, not part of the minimum optimize workflow.

## Pattern References

Do not read all pattern references at once.

Use [patterns/index.md](references/patterns/index.md) to choose the most relevant optimization direction first. Treat that index as the single entry point for detailed pattern references under `references/patterns/`.

## Helper Script Usage

Use the bundled helper script to execute project subcommands from this repository:

```bash
python3 ../scripts/run-command.py gen-test --input <operator.py> --test-mode <mode>
python3 ../scripts/run-command.py run-test --test-file <test.py> --operator-file <candidate.py> --test-mode <mode>
python3 ../scripts/run-command.py run-bench --bench-file <bench.py> --operator-file <candidate.py> --bench-mode <mode>
```

Use the resolved optimize modes when filling `<mode>`. Pass explicit `--output` only when the workflow needs a non-default artifact path.

## Workflow

1. Inspect the operator workspace and confirm the current test and benchmark artifacts.
2. Resolve the correctness mode, defaulting to `differential` unless the task explicitly says otherwise.
3. Resolve the benchmark mode, defaulting to `standalone` unless the task explicitly says otherwise.
4. Generate missing correctness tests through the bundled helper script using the `gen-test` subcommand with the resolved correctness mode:
   `python3 ../scripts/run-command.py gen-test --input <operator.py> --test-mode <mode>`
5. Generate missing benchmark cases through the bundled helper script using the `gen-bench` subcommand with the resolved benchmark mode:
   `python3 ../scripts/run-command.py gen-bench --input <operator.py> --bench-mode <mode>`
6. Treat the original operator as validated candidate `round 0`.
7. Build a candidate pool from validated versions instead of assuming the current best version is always the right parent.
8. For the next optimization attempt, choose one validated parent candidate and record that parent explicitly.
9. Create a new `opt-round-N/` directory and copy the chosen parent operator into that directory before editing it.
10. Create or update `opt-round-N/attempts.md` from the start of the round and append every meaningful attempt, failure, repair, and measurement.
11. Read `references/patterns/index.md` and select one primary optimization pattern for the current round.
12. Read only the one or two detailed pattern references that match the chosen round hypothesis.
13. Apply one coherent optimization theme for the round.
14. Run correctness validation through the bundled helper script using the `run-test` subcommand with the resolved correctness mode:
   `python3 ../scripts/run-command.py run-test --test-file <test.py> --operator-file <candidate.py> --test-mode <mode>`
15. If correctness fails, repair the optimized operator in place, record the failure and repair in `attempts.md`, and re-run `run-test` until it passes or the round direction is no longer viable.
16. After correctness passes, run performance validation through the bundled helper script using the `run-bench` subcommand with the resolved benchmark mode:
   `python3 ../scripts/run-command.py run-bench --bench-file <bench.py> --operator-file <candidate.py> --bench-mode <mode>`
17. If performance regresses or does not improve enough, record the result in `attempts.md`, keep iterating within the same round, or abandon that direction and choose another validated parent for a later round.
18. When a round achieves a real performance win, finalize the round `summary.md`, update `opt-note.md`, and keep the round artifacts intact for reuse.

## Quality Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Always run correctness before trusting performance results.
- Keep parent-child traceability explicit so later engineers can understand which idea produced each round.
- Prefer multiple diverse optimization directions over a single greedy chain from the current best version.
- Prefer selective pattern reading over bulk-loading all optimization references.
- Do not silently discard optimization intent; preserve important comments that explain why a change helps.
- Record within-round attempts continuously so long-running rounds do not lose intermediate learning.
- Record optimization points in enough detail that another engineer could reuse them on a related operator.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
