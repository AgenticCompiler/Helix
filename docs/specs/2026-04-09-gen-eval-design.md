# Gen Eval Design

## Summary

- Add a new `gen-eval` subcommand that runs one code-agent task to repair an operator when needed, generate correctness tests, generate a benchmark, and validate both artifacts before finishing.
- Keep the CLI thin by expressing the workflow in a dedicated staged skill instead of teaching the wrapper to orchestrate multiple generation subcommands itself.
- Stage only the skills required for this workflow so `gen-eval` does not copy the optimize or analysis skill sets into the target workspace.
- Carry `--remote` and `--remote-workdir` through prompt context so the generated validation commands execute remotely when requested.

## Goals

- Let a user point at one operator file and ask the code agent to make the operator runnable and fully covered by generated test and benchmark harnesses.
- Allow the workflow to repair the original operator file directly when failures come from the operator implementation rather than the generated harnesses.
- Reuse the existing generation conventions for output names, test mode selection, bench mode selection, and validation command patterns.
- Keep skill staging explicit so this workflow stays separate from optimize, profiler, and IR-analysis flows.

## Non-Goals

- Do not introduce optimize rounds, `opt-round-*` artifacts, or `opt-note.md`.
- Do not add profiling or IR inspection behavior to this command.
- Do not make the CLI itself sequentially run `gen-test` and `gen-bench` as separate subcommands.
- Do not protect the input operator file from edits; this workflow is intentionally allowed to modify it in place.

## CLI Contract

- Add `gen-eval` as a new agent-backed subcommand.
- `gen-eval` accepts:
  - `--input/-i`
  - `--output/-o`
  - `--agent`
  - `--interact`
  - `--show-output`
  - `--verbose`
  - `--remote`
  - `--remote-workdir`
  - `--test-mode {standalone,differential}`
  - `--bench-mode {standalone,msprof}`
- `--test-mode` defaults to `differential`.
- `--bench-mode` defaults to `standalone`.
- `--output` remains the derived operator output path for agent-backed generation-style commands and should default to `opt_<operator>.py` for prompt context only, even though this workflow edits the original operator in place. The generated test and benchmark outputs continue using the standard derived names:
  - `test_<operator>.py` or `differential_test_<operator>.py`
  - `bench_<operator>.py`

## User-Visible Semantics

- `gen-eval` launches one agent task, not a supervisor loop.
- The workflow may edit the original operator file directly.
- The workflow must generate both a test harness and a benchmark harness before finishing.
- The workflow must execute both generated artifacts before finishing.
- If execution fails:
  - repair the generated harness when the fault belongs to the harness
  - repair the original operator file when the fault belongs to the operator
- If the outer task is remote-aware, every validation command must include the same `--remote` and reuse `--remote-workdir` when provided.
- The command should end with a short assumptions summary like the existing generation commands.

## Skill Contract

- Add a new skill named `eval-gen`.
- `eval-gen` is the workflow entrypoint for `gen-eval`.
- The skill should explicitly require this order:
  1. Inspect the operator file and identify likely operator defects.
  2. Repair the original operator file if there is a clear operator-level problem.
  3. Generate the correctness test through `test-gen`.
  4. Validate the test through `operator-eval`.
  5. Repair the generated test or the original operator depending on the failure source.
  6. Generate the benchmark through `bench-gen`.
  7. Validate the benchmark through `operator-eval`.
  8. Repair the generated benchmark or the original operator depending on the failure source.
  9. Re-run validation after any relevant repair until both artifacts pass or the environment blocks progress.
- The skill should explicitly forbid optimize-only behavior:
  - no `opt-round-*`
  - no `opt-note.md`
  - no profiler-driven investigation
  - no IR-capture flows

## Skill Staging

- `gen-eval` should stage only the minimum skills required for the workflow:
  - `eval-gen`
  - `test-gen`
  - `bench-gen`
  - `operator-eval`
- `gen-eval` should not stage:
  - `optimize`
  - `ascend-npu-operator-profiler`
  - `ascend-operator-ir-analyzer`
- Existing commands keep the current staging behavior unless they explicitly request a restricted staged-skill set.

## Prompt Contract

- The `gen-eval` prompt must include:
  - a short intro that this is a combined test-and-benchmark generation task
  - the operator input path
  - the requested test output path
  - the requested benchmark output path
  - the requested test mode
  - the requested benchmark mode
  - an explicit statement that the workflow may edit the original operator file directly when the operator is at fault
  - an explicit statement that both generated artifacts must be executed before the task finishes
  - an explicit statement that remote-aware validation commands must carry through `--remote` and `--remote-workdir`
- The prompt should explicitly direct the agent to use the staged `eval-gen` skill as the entry workflow.

## Implementation Shape

- Add `CommandKind.GEN_EVAL` and map it to the new `eval-gen` skill.
- Add a dedicated generation-eval handler alongside the existing generation handlers.
- Extend `AgentRequest` with optional staged-skill filtering data so the runtime can request a subset of skills.
- Extend `SkillLinkManager` with a way to copy only named skills while preserving current symlink-safety checks and cleanup behavior.
- Reuse the existing agent runner and result rendering flow instead of adding a second execution path.
- Keep optimize-specific guidance and supervision unchanged; `gen-eval` should not use them.

## Error Handling

- Missing input path should remain a short parser error.
- Skill staging should fail explicitly if a requested staged skill path already exists as a symlink.
- If a requested staged skill name does not exist in the repository `skills/` directory, fail explicitly instead of silently skipping it.
- The workflow may stop when failure is clearly caused by environment issues such as missing modules, missing toolchains, or unreachable remote targets that cannot be fixed from repository code.

## Testing

- Parser coverage for `gen-eval` command mapping and supported flags.
- Prompt coverage for:
  - default test and bench modes
  - explicit operator-repair wording
  - remote propagation wording
  - dual output-path wording
- Handler and runtime coverage for correct `AgentRequest` construction.
- Skill staging coverage proving `gen-eval` stages only the requested subset and leaves excluded skills unstaged.
- Contract coverage proving the new `eval-gen` skill documents direct operator repair and remote-aware validation behavior.
- Full verification with `ruff`, `pyright`, and `unittest`.
