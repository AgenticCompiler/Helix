## Summary

When MCP runtime dependencies such as `fastmcp`, `uvicorn`, or `pydantic` are not
installed, non-MCP entrypoints such as `triton-agent --help` should still import
and run. The CLI should only require those packages when the user explicitly
enables managed MCP support or starts the standalone MCP server.

## Root Cause

`src/triton_agent/run_eval_mcp_server.py` imports MCP server runtime dependencies at
module import time. That module is imported transitively by `triton_agent.mcp`,
which is imported by backend modules during normal CLI startup. As a result, even
read-only commands and help rendering fail before argument handling can decide
whether MCP is needed.

## Design

Move MCP runtime dependency imports behind runtime helpers inside the MCP server
module. Keep module-level utilities, constants, and server lifecycle functions
importable without those packages. Only `create_server()`, request-handling paths,
and HTTP server startup should load the optional dependencies.

## Validation

Add a regression test that blocks `fastmcp`, `uvicorn`, and `pydantic` imports in a
subprocess and verifies `triton_agent.cli.main(["--help"])` exits successfully.
