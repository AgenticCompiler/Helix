# Convert Prompt Option Design

## Summary

- Add `--prompt` to both `convert` and `convert-batch`.
- Make convert prompt handling match `optimize` and `optimize-batch`.
- Apply the same additional user instructions to each batch workspace request.

## Problem

`optimize` and `optimize-batch` already let callers append extra task guidance with `--prompt`.

`convert` and `convert-batch` do not expose the same option, even though the prompt-building layer already has shared support for appending additional user instructions. This makes conversion workflows less consistent and forces users to encode special guidance outside the CLI surface.

## Goals

- Keep `convert` and `convert-batch` aligned with the existing `optimize` prompt contract.
- Reuse the existing prompt-append helper instead of introducing convert-specific prompt formatting.
- Ensure batch conversion forwards the same extra instructions to every workspace request.

## Non-Goals

- Do not change the base convert workflow contract or staged skill set.
- Do not add per-workspace prompt customization for batch conversion.
- Do not change output naming, concurrency, or test-mode behavior.

## User-Facing Behavior

For single conversion:

- `convert --input kernel.py --prompt "Preserve the public function name."` appends that text under an `Additional user instructions:` section after the standard convert prompt.

For batch conversion:

- `convert-batch --input ./workspaces --prompt "Avoid changing numerics."` appends the same instructions to every workspace request built from that batch run.

For omitted or blank values:

- when `--prompt` is not provided, convert behavior stays unchanged
- blank prompt values do not add an empty section to the built prompt

## Design

- Extend the shared generation option model with an optional `prompt` field so both convert commands can carry user instructions through existing convert request construction.
- Expose `--prompt` on the `convert` and `convert-batch` parser definitions, matching the existing `optimize` command UX.
- In convert request construction, build the standard convert prompt first, then append user instructions with the shared prompt helper already used by optimize.

## Validation

Add tests that verify:

- `convert` accepts `--prompt` and stores it in parsed options
- `convert-batch` accepts `--prompt` and stores it in parsed options
- single-request convert appends the additional user instructions section
- batch convert applies the same appended prompt to each workspace request
