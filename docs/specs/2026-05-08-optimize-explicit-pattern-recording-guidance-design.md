# Optimize Explicit Pattern Recording Guidance Design

## Summary

Make the optimize workflow guidance explicitly tell agents to record selected pattern choices in the round markdown artifacts.

Keep this as a visibility and prompt-quality improvement, not a new hard validation rule.

## Goal

Reduce cases where optimize rounds use pattern triage but fail to write down the chosen pattern direction clearly.

## Scope

- Add explicit optimize prompt guidance for recording selected patterns in `attempts.md`.
- Add explicit optimize prompt guidance for recording the final selected pattern direction in `summary.md`.
- Add matching explicit wording near the optimize skill's round-entry and pattern-triage guidance.
- Update focused tests that cover optimize prompt and guidance text.

## Non-Goals

- Do not add new `round-state.json` fields.
- Do not make `check-round` parse markdown content.
- Do not require every round to map to a named pattern.

## Rationale

The current optimize docs already describe where pattern choices belong, but the instruction is easy to miss because it primarily lives in artifact-reference sections.

Putting the reminder directly in the optimize prompt and earlier in the workflow skill makes the expectation visible at the moment the agent is deciding and recording a round direction.

## Validation

- Prompt tests should expect the explicit recording guidance.
- Shared optimize guidance tests should expect the same wording.
- Existing round-gate behavior should remain unchanged.
