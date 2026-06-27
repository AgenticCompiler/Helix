# Stream Output Naming Scope Design

## Summary

- Keep CLI/parser and internal request/option state on `stream_output`.
- Keep compatibility-oriented helper APIs and log naming on `show_output`.
- Avoid broad renames that churn stable helper call sites without user-visible benefit.

## Goals

- Make parser, namespace, and dataclass state read consistently as `stream_output`.
- Preserve stable helper contracts where `show_output` still describes a compatibility or log-facing behavior.
- Limit this change to naming cleanup without changing runtime streaming semantics.

## Non-Goals

- Do not rename `show-output.log` files or `show_output_log.py`.
- Do not rename helper APIs such as `generate_workspace_report(..., show_output=...)` when they already describe report/log behavior cleanly.
- Do not change the default streaming behavior introduced earlier.

## Implementation Notes

- Keep `args.stream_output`, `AgentRequest.stream_output`, `GenerationOptions.stream_output`, `ConvertOptions.stream_output`, and `OptimizeRunOptions.stream_output`.
- Keep `render_result(..., show_output=...)` as the small compatibility boundary used by direct execution commands and existing helper-style call sites.
- Keep report/log-check helper signatures on `show_output`, but map that value into `AgentRequest.stream_output` when constructing requests.

## Testing

- Parser and model tests should assert `stream_output` on args and dataclasses.
- Helper/API tests should keep using `show_output` where the public helper signature intentionally stays unchanged.
- Regression coverage should prove report/log-check helpers still construct working requests after the internal rename.
