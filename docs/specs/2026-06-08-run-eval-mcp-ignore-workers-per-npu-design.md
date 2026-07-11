# Run-Eval MCP Runtime Ignores Workers Per NPU

## Summary

When `--enable-mcp` is active, the shared run-eval MCP server should serialize
device-bound tool execution per physical NPU device, not per expanded
`workers_per_npu` slot.

## User-Visible Semantics

- Keep the existing batch CLI interface unchanged.
- Keep `--concurrency max` behavior unchanged.
- Keep non-MCP batch affinity behavior unchanged.
- In MCP mode, the managed run-eval server leases at most one active tool
  invocation per configured NPU device.
- `HELIX_BATCH_WORKERS_PER_NPU` remains accepted for compatibility, but
  the managed MCP runtime ignores it when building its runtime device pool.

## Design

Update `src/helix/run_eval_mcp_server.py` so the managed device pool is
built directly from the configured device list, with one pool entry per device.
Retain the existing device parsing and default-to-device-`0` behavior for
standalone debugging.

## Validation

- Add regression tests that verify `build_slot_pool("0,1", 2)` returns
  `("0", "1")`.
- Add regression tests that verify `configured_slot_pool()` ignores
  `HELIX_BATCH_WORKERS_PER_NPU` when devices are configured.
