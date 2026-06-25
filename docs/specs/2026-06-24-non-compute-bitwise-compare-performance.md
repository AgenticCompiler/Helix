# Non-Compute Bitwise Compare Performance

## Goal

Reduce the worst-case runtime and allocation cost of `compare-result` for `compute=False` outputs without changing the user-visible comparison semantics.

## Current Behavior

`non-compute` comparisons require exact raw-bit equality after the existing shape and dtype checks. This means:

- identical `NaN` payloads must still compare equal
- `+0.0` and `-0.0` must still compare different

The current implementation achieves this by converting each tensor storage into a Python `bytes` object and comparing the resulting byte strings. That preserves the required semantics, but it forces large tensor payloads through Python object materialization.

## Change

Keep the existing shape, dtype, and raw-bit equality contract, but replace the Python `bytes(...)` storage comparison with a PyTorch-backed byte-tensor equality check:

- make each tensor contiguous
- reinterpret each tensor as `torch.uint8`
- compare the byte views with `torch.equal(...)`

## Non-Goals

- No change to `compute=True` floating-point, integer, or bool comparison behavior.
- No change to payload structure validation or diagnostics wording beyond what is necessary to keep the current messages.
- No opportunistic performance work in other comparison paths as part of this fix.
