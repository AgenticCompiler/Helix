## Summary

When `fastmcp` is not installed, non-MCP entrypoints such as `triton-agent --help`
should still import and run. The CLI should only require `fastmcp` when the user
explicitly enables managed MCP support or starts the standalone MCP server.

## Root Cause

`src/triton_agent/run_eval_mcp_server.py` imports `fastmcp` at module import time.
That module is imported transitively by `triton_agent.mcp`, which is imported by
backend modules during normal CLI startup. As a result, even read-only commands and
help rendering fail before argument handling can decide whether MCP is needed.

## Design

Move the `fastmcp` imports behind runtime helpers inside the MCP server module.
Keep module-level utilities, constants, and server lifecycle functions importable
without `fastmcp`. Only `create_server()` and request-handling paths should load the
optional dependency.

## Validation

Add a regression test that blocks `fastmcp` imports in a subprocess and verifies
`triton_agent.cli.main(["--help"])` exits successfully.
