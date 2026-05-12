# NPU Load Order Optimization Pattern

## Summary

Reorder independent loads so false sequencing does not serialize memory traffic and create avoidable wait in memory-bound kernels.

Ascend lowering generally respects the programmed load order, so dependency-free loads that appear later may miss overlap opportunities if they are placed behind dependent work.

## Use When

- Hot loops contain multiple loads with no true dependency but issue serially.
- Loop-carried dependencies force one load to wait, and unrelated loads are placed after it.
- Profile evidence suggests memory latency dominates more than arithmetic throughput.

## Avoid When

- Load reordering would violate true data dependencies or semantics.
- Kernel is too small for scheduling changes to matter.
- Root bottleneck is not memory sequencing.

## Signals

### Code

- Independent `tl.load` operations appear after dependent pointer resolution.
- Address setup and dependent loads are interleaved so independent loads start late.

### Profile

- Memory-bound behavior persists with low arithmetic sensitivity.
- Small arithmetic changes do not move runtime, but sequencing changes might.

## Optimization Strategy

1. Identify truly independent loads.
2. Move them earlier, before dependent load chains where legal.
3. Keep semantic ordering and loop-carried correctness intact.
4. Validate with parent-vs-child benchmark evidence.

## Example

### Before

```python
idx_B = tl.load(p_B_index)      # depends on previous-iteration store
p_B = B_ptr + idx_B
b_B = tl.load(p_B)              # dependent load
b_A = tl.load(p_A)              # independent load arrives late
```

### After

```python
b_A = tl.load(p_A)              # independent load starts earlier
idx_B = tl.load(p_B_index)      # dependent chain follows
p_B = B_ptr + idx_B
b_B = tl.load(p_B)
```

## Common outcomes

- Small but real wins in stable memory-bound kernels where dependency chains were avoidably serialized.
- Larger gains when change also converts strided/discrete access into contiguous staged loads.
- Flat or noisy results when launch geometry/layout remains the dominant bottleneck.

## Risks

- Reordering can silently break semantics if dependency assumptions are wrong.
- Aggressive motion around stores can introduce loop-carried hazards.
- Cleaner schedule may still lose if another pattern is primary.

## What To Verify After Applying

- Each moved load is independent of operations it crosses.
- Loop-carried behavior and numerical results match reference.
- Benchmark/profile evidence confirms expected overlap or latency reduction.

## Related Patterns

- `loop-invariant-hoisting`
- `tiling`
- `program-multiple-rows`
