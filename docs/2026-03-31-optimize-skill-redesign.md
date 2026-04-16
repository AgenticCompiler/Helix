## Summary

- Redesign the `skills/optimize` skill from a placeholder into a complete iterative optimization workflow for Triton Ascend NPU operators.
- Treat optimization as search over validated candidates instead of a single linear chain from the current best version.
- Require persistent round artifacts and optimization notes so later engineers can reuse successful ideas.

## User-Visible Behavior

- Given an operator file, the skill must first ensure the operator directory has correctness tests and benchmark cases.
- If correctness tests are missing, generate them with `triton-npu-gen-test` in `differential` mode.
- If benchmark cases are missing, generate them with `triton-npu-gen-bench` in `standalone` mode.
- Treat the original operator as validated candidate `round 0`.
- For each optimization round, create a new `opt-round-N/` directory and derive a new candidate from a previously validated parent, not necessarily the current best performer.
- After every code change, run correctness validation with `run-test` before trusting the candidate.
- After correctness passes, run performance validation with `run-bench`.
- Continue repairing or revising the round until the new candidate shows a measurable performance improvement over its chosen parent or the current comparison target.
- Persist each round's optimized operator, performance artifacts, and `summary.md`.
- Append a concise round entry to `opt-note.md`, including the parent round, optimization theme, performance outcome, and a link to the round summary.

## Design Decisions

- Use a candidate-pool search strategy to preserve optimization diversity and avoid overcommitting to a single local optimum.
- Restrict parent selection to validated candidates so every new round starts from code that already passes correctness checks.
- Keep the top-level `SKILL.md` concise and move detailed workflow rules, artifact contracts, and note format into `references/`.
- Keep command templates and remote-command variants out of the top-level `SKILL.md` unless they are essential to triggering or routing; prefer one short pointer to the bundled helper script instead.
- Reuse the existing optimization pattern reference files as optional idea sources after the workflow contract is established.

## Planned Skill Structure

- `skills/triton-npu-optimize/SKILL.md`: trigger guidance, inputs, outputs, primary workflow, and required references
- `skills/triton-npu-optimize/references/workflow.md`: round lifecycle, candidate selection, and recovery rules
- `skills/triton-npu-optimize/references/artifacts.md`: required directory layout and per-round artifacts
- `skills/triton-npu-optimize/references/opt-note-format.md`: expected `opt-note.md` structure and entry content
- `skills/triton-npu-optimize/references/round-failure-handling.md`: round failure handling guidance for correctness failures, benchmark failures, and regressions

## Validation

- Run `quick_validate.py` against `skills/optimize`
- Manually review the resulting `SKILL.md` for concise triggers and clear reference navigation
