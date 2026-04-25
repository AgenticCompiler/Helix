# Generation Prompt Option Design

## Summary

- Add `--prompt` to `gen-eval`, `gen-eval-batch`, `gen-test`, and `gen-bench`.
- Make generation prompt handling match the existing `convert` and `optimize` prompt contract.
- Apply the same additional user instructions to every request built from `gen-eval-batch`.

## Problem

The generation workflows already have a shared prompt-building path, but only `convert` and `optimize` expose a CLI-level `--prompt` option for callers to append task-specific instructions.

This makes generation commands less flexible and less consistent. Users who want to preserve naming, avoid certain harness patterns, or carry special validation guidance into the agent prompt cannot express that through the public CLI surface.

## Goals

- Keep single-workspace generation commands aligned with the existing `convert` and `optimize` UX.
- Reuse the shared prompt append helper instead of creating generation-specific prompt formatting.
- Ensure batch `gen-eval` forwards the same additional instructions to every discovered workspace request.

## Non-Goals

- Do not change the base `gen-eval` workflow contract, staged skills, or validation sequence.
- Do not add per-workspace prompt customization for batch generation.
- Do not change generation output naming, overwrite rules, concurrency, or mode defaults.

## User-Facing Behavior

For single generation commands:

- `gen-test --input kernel.py --prompt "Preserve helper names."` appends that text under an `Additional user instructions:` section after the standard generation prompt.
- `gen-bench --input kernel.py --prompt "Keep benchmark shapes small."` does the same for benchmark generation.
- `gen-eval --input kernel.py --prompt "Avoid broad operator rewrites."` appends the same section to the combined eval-generation prompt.

For batch generation:

- `gen-eval-batch --input ./workspaces --prompt "Avoid changing numerics."` appends the same instructions to every per-workspace request built from that batch run.

For omitted or blank values:

- when `--prompt` is not provided, generation behavior stays unchanged
- blank prompt values do not add an empty section to the built prompt

## Design

- Extend the shared `GenerationOptions` payload with an optional `prompt` field so generation commands can carry user instructions through existing request construction.
- Expose `--prompt` on the parser definitions for `gen-eval`, `gen-eval-batch`, `gen-test`, and `gen-bench`.
- In generation request construction, build the standard prompt first, then append user instructions with `append_additional_user_instructions(...)`.
- Keep batch propagation on shared request construction only; `gen-eval-batch` should not introduce a separate prompt branch.

## Validation

Add tests that verify:

- each supported generation command accepts `--prompt` and stores it in parsed options
- generation request building appends the additional user instructions section
- `gen-eval` keeps its existing staged skill set while carrying the appended prompt
- `gen-eval-batch` applies the same appended prompt to every workspace request
