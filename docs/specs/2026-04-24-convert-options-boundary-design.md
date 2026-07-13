# Convert Options Boundary Design

## Summary

- Replace convert's reuse of `GenerationOptions` with a dedicated `ConvertOptions` type.
- Keep the new `--prompt` behavior unchanged for `convert` and `convert-batch`.
- Make the convert options model contain only fields that belong to convert.

## Problem

`convert` and `convert-batch` currently reuse `GenerationOptions` even though convert does not expose the full generation option surface.

That blurs the ownership boundary between conversion and generation, and it encourages convert-only changes such as `--prompt` to leak into a generation-focused model.

## Goals

- Give convert a feature-local options type with a name that matches its ownership.
- Keep generation models scoped to `gen-*` commands.
- Preserve current convert CLI behavior, including `--prompt`.

## Non-Goals

- Do not change convert prompt semantics.
- Do not change staged skills, batch traversal, or output naming.
- Do not refactor unrelated command families.

## Design

- Add `ConvertOptions` under `src/helix/convert/models.py`.
- Move `convert` and `convert-batch` command parsing, batch execution, and request building to that type.
- Keep only convert-owned fields in `ConvertOptions`: interaction, verbosity, output controls, agent and remote settings, test mode, output path, and optional user prompt.
- Remove the temporary convert-specific `prompt` extension from `GenerationOptions` so generation models return to serving only generation commands.

## Validation

Add or update tests that verify:

- convert parser helpers return `ConvertOptions`
- convert request construction still appends additional user instructions
- batch convert still forwards the same prompt to each workspace request
- generation tests continue using `GenerationOptions` unchanged
