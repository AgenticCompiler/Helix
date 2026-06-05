## Summary

The run-eval MCP server must expose short, explicit metadata for both tool-level and parameter-level descriptions so MCP clients can render usable tool help without relying on skill prose or CLI-style references.

## User-Visible Behavior

When an MCP client lists the run-eval tools, each tool should include a concise description of when to use it. Each input parameter should also include a short description in the JSON schema so clients can show inline help while filling tool arguments.

The MCP tool contract should stay narrower than the legacy CLI surface. Internal execution toggles such as `verbose` and `keep_remote_workdir` should not be exposed as tool parameters.

When a parameter is effectively enum-like but still exposed as a plain string in the schema, its description should spell out the supported values so MCP clients can guide argument entry without reading separate reference docs. This applies to mode selectors and other constrained choice fields such as `metric_source`.

Tool-specific parameter sets should also be narrowed to match the intended workflow instead of mirroring every legacy CLI combination. For example, `run-test-baseline` should not expose differential comparison inputs that belong to optimize or follow-up comparison flows.

## Implementation Notes

Use explicit FastMCP metadata instead of implicit docstring parsing:

- Set each tool description with `@server.tool(..., description=...)`.
- Set each parameter description with `Annotated[..., Field(description=...)]`.
- Omit internal-only CLI flags from the tool function signature so they do not appear in the published MCP schema.
- Include supported value names in parameter descriptions for mode-like and choice-constrained string fields such as `test_mode`, `bench_mode`, and `metric_source`.
- Remove tool inputs that are not part of the intended MCP workflow, even if the underlying script still supports them.

This keeps the MCP contract stable even if docstring parsing behavior changes and makes the schema easy to verify in tests.
