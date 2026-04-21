# Baseline Contract Field Map Design

## Summary

- Replace the duplicated baseline contract JSON fields with one ordered mapping, `baseline_state_fields`.
- Derive required baseline field names from that mapping's keys.
- Keep `round_state_required_fields` unchanged in this small cleanup.

## Goals

- Remove redundant baseline field-name duplication in `skills/triton-npu-optimize-check/references/contract.json`.
- Keep baseline parser behavior unchanged.
- Keep prompt and guidance rendering order stable.

## Non-Goals

- Do not change round contract structure in this change.
- Do not change baseline validation semantics or error messages.

## Design

- Replace:
  - `baseline_state_required_fields`
  - `baseline_state_field_descriptions`
- With:
  - `baseline_state_fields`

The new shape is an ordered JSON object:

```json
{
  "baseline_state_fields": {
    "baseline_kind": "description",
    "source_operator": "description"
  }
}
```

Readers should:

- treat `baseline_state_fields.keys()` as the required baseline fields
- treat `baseline_state_fields.items()` as the prompt-rendering source

## Expected Outcome

- One source of truth for baseline field names and descriptions.
- No behavior change outside the contract JSON shape.
