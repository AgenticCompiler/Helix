# Optimize Round-Local Output Path Design

## Summary

- Remove the root-level optimize output-path hint from optimize-specific prompts.
- Make the optimize round contract the only output-location instruction for optimized operator snapshots.
- Keep optimize artifacts round-local: `workdir/opt-round-N/opt_<operator>.py`.
- Preserve internal request compatibility for `output_path` so non-prompt optimize plumbing does not need a broader refactor in this change.

## Problem

The current optimize flow carries a generic `output_path` field whose default value resolves to `workdir/opt_<operator>.py`.

That path is acceptable as an internal request field, but it becomes misleading when optimize prompts surface it to the agent as:

- `Requested output: workdir/opt_<operator>.py`

Later in the same optimize guidance, the prompt and the staged optimize skill already require round artifacts to live inside `opt-round-N/`.

This creates a prompt-level contradiction:

- one part implies the optimized file should be written at the workspace root
- another part requires the optimized file to be written inside the current round directory

In practice, this can cause the agent to generate `workdir/opt_<operator>.py` even though the real optimize artifact contract requires `workdir/opt-round-N/opt_<operator>.py`.

## Goals

- Eliminate prompt guidance that suggests root-level optimize outputs.
- Keep the round-local artifact contract explicit and unambiguous.
- Avoid a broad runtime or CLI refactor in this fix.
- Update tests so they enforce the prompt contract we actually want.

## Non-Goals

- Do not remove the `output_path` field from `AgentRequest` in this change.
- Do not redesign non-optimize commands that legitimately use `Requested output: ...`.
- Do not change the round artifact naming convention.
- Do not change optimize resume, baseline, or batch-loop semantics beyond prompt wording.
- Do not remove the optimize CLI `--output` flag in this change.

## User-Facing Behavior

### Optimize Worker Prompt

Optimize worker prompts should no longer mention a root-level requested output path.

Instead, the prompt should rely on the existing round artifact rules, especially:

- each round writes the optimized operator snapshot as `opt_<original-operator>.py`
- that snapshot lives inside the current `opt-round-N/`

The intended effective output location for a round is therefore:

- `workdir/opt-round-N/opt_<operator>.py`

### Optimize Baseline Prompt

Optimize baseline-repair prompts should also stop mentioning a requested root-level optimize output path.

Baseline setup is responsible for repairing or establishing `baseline/`, not for instructing the agent to write the optimized round artifact at the workspace root.

The baseline prompt may still include:

- operator input
- requested test mode
- requested bench mode
- remote execution context
- target chip and optimize target

## Implementation Notes

### Prompt Scope

This change should be implemented in optimize-specific prompt builders, not by weakening shared prompt behavior for unrelated commands.

That means:

- keep the generic shared prompt builder behavior for commands like `gen-test`, `gen-bench`, and `convert`
- special-case optimize so root-level requested output paths are not surfaced in optimize prompts

### Compatibility Strategy

Keep the current internal `output_path` request field for now.

Rationale:

- optimize runtime and tests already pass this field around
- the behavior bug is caused by prompt wording, not by a demonstrated runtime write requirement
- removing the field or CLI option would be a larger contract change than needed for this fix

## Testing

Update prompt and optimize runtime tests so they verify:

- optimize prompts do not include `Requested output: ...`
- optimize baseline prompts do not include `Requested output: ...`
- round-local optimize artifact instructions remain present

## Acceptance Criteria

- No optimize-specific prompt tells the agent to write `workdir/opt_<operator>.py`.
- Optimize guidance still clearly requires `opt-round-N/opt_<operator>.py`.
- Existing non-optimize prompt tests continue to cover explicit requested-output behavior where that behavior is still correct.
