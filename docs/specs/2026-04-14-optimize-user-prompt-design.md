# Optimize User Prompt Design

## Summary

- Add a `--prompt TEXT` option to both `optimize` and `optimize-batch`.
- Treat the value as additive guidance that is appended to the default optimize worker prompt instead of replacing built-in optimize instructions.
- Preserve the appended guidance across optimize resume and continue flows by carrying it inside the request prompt.
- Keep supervisor prompts unchanged when `--supervise on` so user-supplied optimization preferences do not affect gate decisions.

## Problem

- The current optimize CLI always launches agents with only the built-in optimize prompt contract.
- Users sometimes need to steer the optimize workflow with task-specific guidance such as preferred hypotheses, constraints, or areas of focus.
- Today there is no explicit optimize option for passing that extra instruction through the CLI, so users must modify skills or prompts indirectly, which is heavier and less reproducible than a command-line flag.

## Goals

- Let users pass extra optimize guidance directly from the CLI.
- Keep the default optimize workflow contract intact so artifact, baseline, and evidence requirements are not weakened.
- Make single-workspace and batch optimize behave the same way by applying the same extra prompt text to each workspace request.
- Ensure continuation flows keep the extra guidance so later rounds do not silently lose the user's intent.

## Non-Goals

- Do not replace the built-in optimize prompt with user text.
- Do not add a separate `--prompt-file` option in this change.
- Do not inject user prompt text into supervisor-only audit passes.
- Do not expand this option to non-optimize commands.

## Approaches Considered

### Recommended: Additive `--prompt TEXT`

- Add one string-valued CLI option.
- Append its content to the generated optimize worker prompt under a fixed heading.
- Treat blank or whitespace-only input as absent.

Why this is the best fit:

- It matches the requested behavior exactly.
- It keeps the CLI surface small and easy to explain.
- It composes naturally with the existing prompt-building and resume flow.

### Alternative: `--prompt-file PATH`

- Read extra instructions from a file instead of the command line.

Why not choose this now:

- It solves a different ergonomics problem than the one requested.
- It adds file I/O and validation behavior that is unnecessary for the first version.

### Alternative: Replace The Default Optimize Prompt

- Use user text as the primary prompt body.

Why not choose this now:

- It would bypass built-in optimize constraints that the CLI is expected to enforce consistently.
- It would make resume and supervised flows harder to reason about.

## User-Facing Design

### CLI Option

Add `--prompt TEXT` to:

- `optimize`
- `optimize-batch`

Examples:

```bash
uv run helix optimize --input operator.py --prompt "Prioritize memory-coalescing improvements."
uv run helix optimize-batch --input operators_root --prompt "Avoid changing numerics unless correctness requires it."
```

### Prompt Layout

When `--prompt` is provided with a non-blank value, append this block to the generated optimize worker prompt:

```text
Additional user instructions:
<user prompt>
```

This block should appear after the built-in optimize instructions so the default contract stays visible and intact.

## Behavior

### Initial Optimize Launch

- `optimize` appends the extra user instructions to the initial worker prompt.
- `optimize-batch` appends the same extra user instructions independently to each workspace request.

### Resume And Continue

- The existing optimize recovery and minimum-round continuation paths already rebuild prompts from `request.prompt`.
- Because the appended user instructions live inside `request.prompt`, they automatically persist across:
  - stall recovery resumes
  - successful-but-incomplete `min_rounds` resumes
  - supervised gate-driven continue prompts

No separate persistence field is required for this change unless a later refactor wants one for clarity.

### Supervised Optimize

When `--supervise on`:

- worker prompts include the appended user instructions
- supervisor prompts do not include the appended user instructions

This preserves the intended separation where workers perform optimization work while supervisors audit artifacts and gate continuation based on existing facts.

### Empty Input

- `None`, `""`, or whitespace-only `--prompt` values should be treated as "not provided".
- In those cases, do not append the `Additional user instructions` section.

## Architecture

### CLI Parsing

- Extend optimize-specific parser configuration so `optimize` and `optimize-batch` both accept `--prompt`.
- Store the parsed value on `OptimizeRunOptions`.

### Prompt Construction

- Add a small helper in the prompt-building layer that appends the `Additional user instructions` section only when needed.
- Use that helper when building optimize requests so both single-workspace and batch optimize share the same formatting.

### Resume Flow Compatibility

- Keep `build_optimize_resume_prompt()` behavior centered on `base_prompt`.
- Do not add special-case logic for user prompts there; preserving the enriched base prompt is sufficient.

### Supervisor Isolation

- Keep `build_optimize_supervisor_prompt()` unchanged.
- The runtime should continue constructing supervisor prompts from dedicated supervisor guidance only.

## Testing

- Add CLI tests that verify `optimize` and `optimize-batch` accept `--prompt`.
- Add optimize runtime or prompt-construction tests that verify:
  - appended prompt text appears in single-workspace optimize requests
  - appended prompt text appears in batch workspace requests
  - blank prompt text does not produce an empty extra section
  - supervised continue prompts preserve the appended worker instructions
  - supervisor prompts remain free of appended user instructions

## Documentation

Update `README.md` optimize examples to mention `--prompt` as additive optimize guidance for both single-workspace and batch runs.

## Expected Outcome

- Users can steer optimize runs with task-specific instructions without modifying skills or backend internals.
- The default optimize contract remains authoritative.
- Resume, batch, and supervised optimize flows all preserve the same user intent consistently where it belongs: the worker prompt.
