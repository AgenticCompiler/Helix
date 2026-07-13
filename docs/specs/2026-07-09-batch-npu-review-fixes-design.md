# Batch NPU Review Fixes

## Summary

Apply the post-merge review fixes for batch NPU affinity without changing the
approved managed MCP contract that `workers-per-npu` is accepted and validated
but ignored for runtime leasing.

## Chosen Direction

- Keep issue 1 as-is: managed MCP still leases one active tool invocation per
  physical configured device.
- Fix only review items 2, 3, and 5.

## User-Visible Semantics

- `run-eval-mcp-server` defaults to device `0` only when no CLI option and no
  legacy environment variable provide a device list.
- An explicit empty `--npu-devices` value, or an explicitly empty
  `HELIX_BATCH_NPU_DEVICES`, is invalid everywhere.
- In MCP-enabled batch flows, `--concurrency max` and capacity validation use
  physical device count, not `workers-per-npu`.
- Managed MCP scope reuse treats semantically equivalent affinity strings such
  as `0,1` and `0, 1` as the same configuration.

## Implementation Notes

- Tighten `configured_slot_pool(...)` so omission and explicit empty values are
  handled differently.
- Canonicalize affinity inputs inside `managed_mcp_scope(...)` before storing
  and comparing active scope state.
- Reuse the existing batch affinity helpers so MCP mode still validates
  `workers-per-npu` input while ignoring it for managed capacity.

## Non-Goals

- Expanding managed MCP runtime leasing by `workers-per-npu`.
- Refactoring the duplicated affinity resolution paths called out in review item
  4.
