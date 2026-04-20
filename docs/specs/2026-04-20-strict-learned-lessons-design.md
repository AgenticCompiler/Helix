# Strict Learned Lessons Design

## Goal

Make `learned_lessons.md` a strict reusable optimization-knowledge log instead of a general optimize-session notebook.

## Problem

The optimize skill currently asks workers to record reusable knowledge, but it does not define a strong admission test. In practice, code agents can interpret this as permission to write round-specific failures, command notes, local operator details, or ordinary progress summaries into `learned_lessons.md`.

That weakens the artifact's value. The intended use is to distill rules that can later become skill guidance, profiling-analysis rules, IR-analysis rules, or optimize pattern references.

## User-Visible Semantics

`learned_lessons.md` should accept only lessons that satisfy all of these criteria:

- The lesson generalizes to a family of Triton Ascend NPU operators.
- The lesson is supported by correctness, benchmark, profiler, IR, or compiler-error evidence.
- The lesson is written as a reusable rule, diagnostic mapping, or optimization heuristic.
- The lesson states the condition where it applies, or the limitation that prevents overuse.
- The lesson could plausibly be promoted into an optimize skill, profiling analysis reference, IR analysis reference, or pattern reference.

The artifact must not receive ordinary round narrative. Round-local attempts, command failures, failed guesses, file names, shape-specific details, and temporary troubleshooting notes belong in `opt-round-N/attempts.md`, `opt-round-N/summary.md`, or `opt-note.md`.

## Implementation Approach

Keep the change in the workflow contract and prompt layer:

- Update `skills/triton-npu-optimize/SKILL.md` to define strict admission criteria and anti-examples.
- Update `skills/triton-npu-optimize/references/artifacts.md` to describe the stricter artifact contract.
- Update optimize worker, unsupervised, and resume prompts in `src/triton_agent/prompts.py` so the rule is visible at execution time.
- Update text contract tests so future edits keep the strict learned-lessons boundary intact.

Do not add a new CLI subcommand or runtime validator. The current issue is prompt semantics, and enforcing lesson quality mechanically would require semantic judgment that is better handled by the code agent workflow for now.

## Testing

Add contract tests that verify:

- The optimize skill documents strict admission criteria for `learned_lessons.md`.
- The optimize artifact reference forbids round-local narrative in `learned_lessons.md`.
- Worker, unsupervised, and resume prompts all mention the strict boundary.

Then run the focused text and prompt test files.
